import argparse
import copy
import math
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from histopathology.extract_features import get_models


PAPER_MERGE_PLANS = {
    "H0-mini": {
        5: [(0, 2), (3, 5), (6, 8), (9, 10), (11, 11)],
        4: [(0, 2), (3, 6), (7, 9), (10, 11)],
    },
    "hibou": {
        4: [(0, 1), (2, 4), (5, 7), (8, 11)],
        3: [(0, 2), (3, 6), (7, 11)],
    },
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def get_blocks(model):
    """Return the transformer block container for the supported backbones."""
    return model.encoder.layer if hasattr(model, "encoder") else model.blocks


def set_blocks(model, blocks):
    """Replace the transformer blocks while preserving the backbone family."""
    if hasattr(model, "encoder"):
        model.encoder.layer = nn.ModuleList(blocks)
    else:
        model.blocks = nn.Sequential(*blocks)


def embed_images(model, images):
    """Run the patch/token embedding stem without executing transformer blocks."""
    if hasattr(model, "embeddings"):
        return model.embeddings(images)
    x = model.patch_embed(images)
    if hasattr(model, "_pos_embed"):
        x = model._pos_embed(x)
    elif hasattr(model, "pos_embed"):
        x = x + model.pos_embed
    return x


def forward_features(model, images):
    """Return pre-classification backbone features for distillation."""
    if hasattr(model, "forward_features"):
        return model.forward_features(images)
    output = model(images)
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, tuple):
        return output[0]
    return output


def freeze_stem(model):
    """Freeze the embedding stem during end-to-end student distillation."""
    if hasattr(model, "embeddings"):
        model.embeddings.requires_grad_(False)
        return
    model.patch_embed.requires_grad_(False)
    if hasattr(model, "pos_embed"):
        model.pos_embed.requires_grad_(False)
    if hasattr(model, "cls_token"):
        model.cls_token.requires_grad_(False)


def choose_micro_batch_size(device):
    """Choose a conservative micro-batch size from available CUDA memory."""
    if device.type != "cuda":
        return 64
    props = torch.cuda.get_device_properties(device)
    mem_gb = props.total_memory / (1024 ** 3)
    if mem_gb > 70:
        return 1024
    if mem_gb > 38:
        return 512
    return 256


class PandaTileDataset(Dataset):
    """Read all prepared foreground tiles from PANDA training-slide tar archives."""

    def __init__(self, csv_path, tar_root, transform):
        """Index image members from the tar archives listed by the PANDA training split."""
        self.transform = transform
        self.tar_root = Path(tar_root)
        frame = pd.read_csv(csv_path)
        if "FILENAME" not in frame.columns:
            raise ValueError("PANDA split CSV must contain a FILENAME column.")

        self.samples = []
        for value in frame["FILENAME"].dropna().astype(str).unique():
            slide_id = Path(value).stem
            tar_path = self.tar_root / f"{slide_id}.tar"
            if not tar_path.exists():
                matches = sorted(self.tar_root.glob(f"{slide_id}*.tar"))
                if not matches:
                    raise FileNotFoundError(f"No tar archive found for PANDA slide {slide_id}")
                tar_path = matches[0]

            with tarfile.open(tar_path, "r") as archive:
                for member in archive.getmembers():
                    if member.isfile() and Path(member.name).suffix.lower() in IMAGE_SUFFIXES:
                        self.samples.append((str(tar_path), member.name))

        if not self.samples:
            raise RuntimeError("No image tiles were found in the PANDA training archives.")
        self._tar_handles = {}

    def __len__(self):
        """Return the number of indexed training tiles."""
        return len(self.samples)

    def __getitem__(self, index):
        """Load and augment one tile."""
        tar_path, tile_name = self.samples[index]
        archive = self._tar_handles.get(tar_path)
        if archive is None:
            if len(self._tar_handles) >= 128:
                _, old_archive = self._tar_handles.popitem()
                old_archive.close()
            archive = tarfile.open(tar_path, "r")
            self._tar_handles[tar_path] = archive

        member = archive.getmember(tile_name)
        file_obj = archive.extractfile(member)
        if file_obj is None:
            raise RuntimeError(f"Could not read {tile_name} from {tar_path}")
        image = Image.open(BytesIO(file_obj.read())).convert("RGB")
        return self.transform(image)


def get_train_transforms(mean, std):
    """Return the augmentation and normalisation pipeline used for histopathology distillation."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def make_average_candidate(teacher_blocks, indices):
    """Create the parameter-averaged surrogate candidate used during auditioning."""
    block = copy.deepcopy(teacher_blocks[indices[0]])
    source_params = [dict(teacher_blocks[i].named_parameters()) for i in indices]
    with torch.no_grad():
        for name, parameter in block.named_parameters():
            parameter.copy_(source_params[0][name])
            for source in source_params[1:]:
                parameter.add_(source[name])
            if not any(token in name.lower() for token in ("gamma", "layerscale", "ls")):
                parameter.div_(len(indices))
    return block


def audition_surrogate_block(teacher_model, indices, loader, device, train_batches=50, eval_batches=50):
    """Fine-tune candidate teacher layers locally and return the lowest-error surrogate."""
    if len(indices) == 1:
        return copy.deepcopy(get_blocks(teacher_model)[indices[0]])

    start_idx, end_idx = indices[0], indices[-1]
    teacher_blocks = get_blocks(teacher_model)
    candidates = [{"name": "Average", "block": make_average_candidate(teacher_blocks, indices)}]
    candidates.extend(
        {"name": f"Layer_{i}", "block": copy.deepcopy(teacher_blocks[i])}
        for i in indices
    )

    for candidate in candidates:
        candidate["block"].to(device).train()
        candidate["optimizer"] = optim.AdamW(candidate["block"].parameters(), lr=2e-3)
        candidate["eval_loss"] = 0.0

    def teacher_mapping(inputs):
        """Return the teacher input and output at the boundaries of the candidate span."""
        with torch.no_grad():
            x = embed_images(teacher_model, inputs)
            for i in range(start_idx):
                x = teacher_blocks[i](x)
                if isinstance(x, tuple):
                    x = x[0]
            input_features = x.detach()

            target = input_features
            for i in range(start_idx, end_idx + 1):
                target = teacher_blocks[i](target)
                if isinstance(target, tuple):
                    target = target[0]
            return input_features, target.detach()

    teacher_model.eval()
    criterion = nn.MSELoss()
    loader_iter = iter(loader)

    def next_batch():
        """Return the next calibration batch, restarting the loader when necessary."""
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    for _ in range(train_batches):
        inputs = next_batch().to(device, non_blocking=True)
        input_features, target_features = teacher_mapping(inputs)
        for candidate in candidates:
            optimizer = candidate["optimizer"]
            optimizer.zero_grad()
            output = candidate["block"](input_features)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target_features)
            loss.backward()
            optimizer.step()

    for candidate in candidates:
        candidate["block"].eval()

    with torch.no_grad():
        for _ in range(eval_batches):
            inputs = next_batch().to(device, non_blocking=True)
            input_features, target_features = teacher_mapping(inputs)
            for candidate in candidates:
                output = candidate["block"](input_features)
                if isinstance(output, tuple):
                    output = output[0]
                candidate["eval_loss"] += criterion(output, target_features).item()

    for candidate in candidates:
        candidate["eval_loss"] /= eval_batches
    candidates.sort(key=lambda item: item["eval_loss"])

    print("Candidate evaluation losses:", flush=True)
    for candidate in candidates:
        print(f"  {candidate['name']:<12} {candidate['eval_loss']:.6f}", flush=True)
    print(f"Selected {candidates[0]['name']}", flush=True)
    return candidates[0]["block"].cpu()


class ActivationCatcher:
    """Capture named intermediate activations for boundary-level deep supervision."""

    def __init__(self):
        """Initialise activation and hook storage."""
        self.activations = {}
        self.hooks = []

    def register(self, module, name, detach=False):
        """Register a forward hook for one student or teacher boundary."""
        def hook(_module, _inputs, output):
            """Store one hooked activation."""
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach() if detach else output

        self.hooks.append(module.register_forward_hook(hook))

    def clear(self):
        """Discard activations from the previous forward pass."""
        self.activations.clear()

    def remove(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def distil_student(student_model, teacher_model, dataloader, device, merge_plan, epochs, batch_size, max_images=None):
    """Optimise the assembled TWT student with boundary and final-feature MSE supervision."""
    if max_images is not None:
        epochs = max(epochs, math.ceil(max_images / len(dataloader.dataset)))

    student_model.to(device).train()
    teacher_model.to(device).eval()
    freeze_stem(student_model)

    student_blocks = get_blocks(student_model)
    teacher_blocks = get_blocks(teacher_model)
    teacher_catcher = ActivationCatcher()
    student_catcher = ActivationCatcher()

    for i, (_, teacher_end) in enumerate(merge_plan):
        teacher_catcher.register(teacher_blocks[teacher_end], f"block_{i}", detach=True)
        student_catcher.register(student_blocks[i], f"block_{i}")

    optimizer = optim.AdamW(student_model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.MSELoss()
    accumulation_steps = max(1, 1024 // batch_size)
    total_images = 0

    try:
        for epoch in range(epochs):
            optimizer.zero_grad()
            running_loss = 0.0
            for step, images in enumerate(dataloader, 1):
                if max_images is not None and total_images >= max_images:
                    break
                images = images.to(device, non_blocking=True)
                total_images += images.shape[0]
                teacher_catcher.clear()
                student_catcher.clear()

                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    with torch.no_grad():
                        teacher_final = forward_features(teacher_model, images)
                    student_final = forward_features(student_model, images)

                    loss = criterion(student_final, teacher_final)
                    for name, student_output in student_catcher.activations.items():
                        loss = loss + criterion(student_output, teacher_catcher.activations[name])
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                if step % accumulation_steps == 0 or step == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * accumulation_steps

            scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs}: distillation loss={running_loss / max(len(dataloader), 1):.6f}", flush=True)
            if max_images is not None and total_images >= max_images:
                break
    finally:
        teacher_catcher.remove()
        student_catcher.remove()

    return student_model


def extract_normalisation(transform):
    """Read mean and standard deviation from a torchvision-style transform pipeline."""
    for operation in transform.transforms:
        if hasattr(operation, "mean") and hasattr(operation, "std"):
            return list(operation.mean), list(operation.std)
    raise RuntimeError("Could not find a Normalize transform for the selected teacher model.")


def main():
    """Audition surrogate layers, distil the TWT student, and save its checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=["H0-mini", "hibou"], default="H0-mini")
    parser.add_argument("--target-depth", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing prepared PANDA tar archives.")
    parser.add_argument("--train-csv", type=Path, required=True, help="PANDA training split CSV.")
    parser.add_argument("--h0-model", type=str, default=None)
    parser.add_argument("--hibou-model-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch size; chosen from GPU memory if omitted.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if args.target_depth not in PAPER_MERGE_PLANS[args.teacher]:
        available = sorted(PAPER_MERGE_PLANS[args.teacher])
        raise ValueError(f"Paper merge plans for {args.teacher} are available at depths {available}.")
    merge_plan = PAPER_MERGE_PLANS[args.teacher][args.target_depth]

    device = torch.device(args.device)
    batch_size = args.batch_size or choose_micro_batch_size(device)

    if args.teacher == "H0-mini" and not args.h0_model:
        raise ValueError("--h0-model is required for H0-mini.")
    if args.teacher == "hibou" and not args.hibou_model_dir:
        raise ValueError("--hibou-model-dir is required for Hibou-B.")

    models = get_models(device, h0_model_path=args.h0_model, hibou_model_dir=args.hibou_model_dir)
    teacher, teacher_transform, _ = models[args.teacher]
    teacher.eval()
    mean, std = extract_normalisation(teacher_transform)

    dataset = PandaTileDataset(args.train_csv, args.root, get_train_transforms(mean, std))
    audition_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    distillation_loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    teacher_blocks = get_blocks(teacher)
    if args.resume:
        new_blocks = [copy.deepcopy(teacher_blocks[start]) for start, _ in merge_plan]
    else:
        new_blocks = []
        for start, end in merge_plan:
            new_blocks.append(
                audition_surrogate_block(teacher, list(range(start, end + 1)), audition_loader, device)
            )

    student = copy.deepcopy(teacher)
    set_blocks(student, new_blocks)
    if args.resume:
        student.load_state_dict(torch.load(args.resume, map_location="cpu"), strict=True)

    student = distil_student(
        student,
        teacher,
        distillation_loader,
        device,
        merge_plan,
        epochs=args.epochs,
        batch_size=max(1, batch_size // 2),
        max_images=args.max_images,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{args.teacher}_depth{args.target_depth}.pth"
    torch.save(student.state_dict(), checkpoint)
    print(f"Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
