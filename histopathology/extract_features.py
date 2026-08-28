import argparse
import json
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def get_blocks(model):
    """Return the transformer block container for the supported backbone families."""
    return model.encoder.layer if hasattr(model, "encoder") else model.blocks


def set_blocks(model, blocks):
    """Replace the transformer block container for the supported backbone families."""
    if hasattr(model, "encoder"):
        model.encoder.layer = nn.ModuleList(blocks)
    else:
        model.blocks = nn.Sequential(*blocks)


class TarTileDataset(Dataset):
    """Read all image members from one prepared foreground-tile tar archive."""

    def __init__(self, tar_path, transform):
        """Index image members and store the backbone transform."""
        self.tar_path = Path(tar_path)
        self.transform = transform
        self._archive = None
        with tarfile.open(self.tar_path, "r") as archive:
            self.tile_names = [
                member.name
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).suffix.lower() in IMAGE_SUFFIXES
            ]

    def __len__(self):
        """Return the number of retained tiles in the archive."""
        return len(self.tile_names)

    def __getitem__(self, index):
        """Decode and transform one tile."""
        if self._archive is None:
            self._archive = tarfile.open(self.tar_path, "r")
        tile_name = self.tile_names[index]
        member = self._archive.getmember(tile_name)
        file_obj = self._archive.extractfile(member)
        if file_obj is None:
            raise RuntimeError(f"Could not read {tile_name} from {self.tar_path}")
        image = Image.open(BytesIO(file_obj.read())).convert("RGB")
        return tile_name, self.transform(image)


def load_h0_mini(model_path, device):
    """Load H0-mini from a local weight file and its neighbouring config.json."""
    model_path = Path(model_path)
    config_path = model_path.parent / "config.json"
    with open(config_path, "r") as file_obj:
        config = json.load(file_obj)

    architecture = config.get("architecture", "vit_base_patch14_reg4_dinov2")
    model_args = config.get("model_args", {})
    model = timm.create_model(
        architecture,
        pretrained=False,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        **model_args,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    pretrained_cfg = config.get("pretrained_cfg", {})
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_cfg["mean"], std=pretrained_cfg["std"]),
    ])
    return model, transform


def load_hibou_b(model_dir, device):
    """Load Hibou-B and its image processor from a local Hugging Face model directory."""
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    return model, transform


def get_models(device, h0_model_path=None, hibou_model_dir=None):
    """Load whichever unpruned histopathology backbones were explicitly configured."""
    models = {}
    if h0_model_path:
        models["H0-mini"] = load_h0_mini(h0_model_path, device) + (Path("."),)
    if hibou_model_dir:
        models["hibou"] = load_hibou_b(hibou_model_dir, device) + (Path("."),)
    return models


def load_extraction_model(args, device):
    """Load the uncompressed backbone or instantiate a TWT student and restore its checkpoint."""
    if args.base_model == "H0-mini":
        if not args.h0_model:
            raise ValueError("--h0-model is required when --base-model H0-mini is selected.")
        model, transform = load_h0_mini(args.h0_model, device)
    else:
        if not args.hibou_model_dir:
            raise ValueError("--hibou-model-dir is required when --base-model hibou is selected.")
        model, transform = load_hibou_b(args.hibou_model_dir, device)

    if args.method == "twt":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --method twt is selected.")
        blocks = list(get_blocks(model))
        set_blocks(model, blocks[: args.depth])
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model_name = f"{args.base_model}_custom_depth_model"
    else:
        model_name = f"{args.base_model}_base_model"

    model.to(device).eval()
    return model, transform, model_name


@torch.no_grad()
def model_forward_and_flatten(model, images):
    """Convert token outputs to the CLS-plus-mean-patch representation used for downstream MIL."""
    output = model(images)
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    elif isinstance(output, tuple):
        output = output[0]

    if output.ndim == 3 and output.size(1) == 261:
        cls_token = output[:, 0]
        patch_tokens = output[:, 5:]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
    return output


def extract_tar_features(tar_path, model, transform, output_file, batch_size, num_workers, device):
    """Extract and save all frozen tile embeddings for one slide tar archive."""
    if output_file.exists():
        print(f"[skip] {output_file}", flush=True)
        return

    dataset = TarTileDataset(tar_path, transform)
    if not dataset:
        print(f"[warn] no image tiles in {tar_path}", flush=True)
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    tile_names = []
    feature_batches = []
    for names, images in loader:
        images = images.to(device, non_blocking=True)
        features = model_forward_and_flatten(model, images)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError(
                f"Expected a 2-D feature tensor for MIL, got {type(features).__name__} "
                f"with shape {getattr(features, 'shape', None)}"
            )
        tile_names.extend(names)
        feature_batches.append(features.cpu())

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".pt.tmp")
    torch.save({"tile_names": tile_names, "features": torch.cat(feature_batches, dim=0)}, temporary)
    temporary.replace(output_file)
    print(f"[saved] {output_file} ({len(tile_names)} tiles)", flush=True)


def resolve_tar(tar_root, slide_id):
    """Resolve a slide identifier to its prepared tar archive."""
    slide_id = Path(str(slide_id)).stem
    exact = tar_root / f"{slide_id}.tar"
    if exact.exists():
        return exact
    matches = sorted(tar_root.glob(f"{slide_id}*.tar"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No tar archive found for {slide_id} in {tar_root}")


def process_panda(args, model, transform, model_name, device):
    """Extract features for PANDA train, validation, and test slides."""
    output_dir = args.out_dir / "panda_extracted_features" / model_name
    for split in ("train", "val", "test"):
        csv_path = args.panda_splits_dir / f"{split}.csv"
        frame = pd.read_csv(csv_path)
        if "FILENAME" not in frame.columns:
            raise ValueError(f"{csv_path} must contain a FILENAME column.")
        for slide_id in frame["FILENAME"].dropna().astype(str).unique():
            tar_path = resolve_tar(args.panda_root, slide_id)
            output_file = output_dir / f"{Path(slide_id).stem}.pt"
            extract_tar_features(
                tar_path, model, transform, output_file,
                args.batch_size, args.num_workers, device,
            )


def process_tar_directory(tar_dir, dataset_name, args, model, transform, model_name, device):
    """Extract features for every tar archive in a prepared TCGA-PRAD or CAMELYON directory."""
    tar_paths = sorted(Path(tar_dir).glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No .tar archives found in {tar_dir}")

    output_dir = args.out_dir / f"{dataset_name}_extracted_features" / model_name
    for tar_path in tar_paths:
        output_file = output_dir / f"{tar_path.stem}.pt"
        extract_tar_features(
            tar_path, model, transform, output_file,
            args.batch_size, args.num_workers, device,
        )


def get_args():
    """Parse dataset, model, checkpoint, and feature-extraction options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-panda", action="store_true")
    parser.add_argument("--run-tcga", action="store_true")
    parser.add_argument("--run-camelyon", choices=["camelyon16", "camelyon17", "none"], default="none")

    parser.add_argument("--panda-root", type=Path)
    parser.add_argument("--panda-splits-dir", type=Path)
    parser.add_argument("--tcga-root", type=Path)
    parser.add_argument("--camelyon-root", type=Path)
    parser.add_argument("--camelyon-tar-subdir", default="tar_tiles_20x_256_otsuTH80")

    parser.add_argument("--base-model", choices=["H0-mini", "hibou"], default="H0-mini")
    parser.add_argument("--h0-model", type=str, default=None)
    parser.add_argument("--hibou-model-dir", type=str, default=None)
    parser.add_argument("--method", choices=["base", "twt"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--depth", type=int, default=6)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("./features"))
    return parser.parse_args()


def main():
    """Extract frozen tile embeddings for the requested downstream cohorts."""
    args = get_args()
    device = torch.device(args.device)
    model, transform, model_name = load_extraction_model(args, device)

    if args.run_panda:
        if args.panda_root is None or args.panda_splits_dir is None:
            raise ValueError("--run-panda requires --panda-root and --panda-splits-dir.")
        process_panda(args, model, transform, model_name, device)

    if args.run_tcga:
        if args.tcga_root is None:
            raise ValueError("--run-tcga requires --tcga-root.")
        process_tar_directory(args.tcga_root, "tcga_prad", args, model, transform, model_name, device)

    if args.run_camelyon != "none":
        if args.camelyon_root is None:
            raise ValueError("--run-camelyon requires --camelyon-root.")
        tar_dir = args.camelyon_root / args.camelyon_tar_subdir
        process_tar_directory(tar_dir, args.run_camelyon, args, model, transform, model_name, device)


if __name__ == "__main__":
    main()
