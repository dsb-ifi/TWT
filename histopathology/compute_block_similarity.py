import argparse
import random
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from histopathology.extract_features import get_blocks, get_models

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class RandomTilesDataset(Dataset):
    """Load a fixed sample of tiles from per-slide tar archives."""

    def __init__(self, tile_refs, transform):
        """Store sampled tile references and the backbone image transform."""
        self.tile_refs = sorted(tile_refs, key=lambda item: str(item[0]))
        self.transform = transform
        self._archive = None
        self._archive_path = None

    def __len__(self):
        """Return the number of sampled tiles."""
        return len(self.tile_refs)

    def __getitem__(self, index):
        """Decode and transform one sampled tile."""
        tar_path, tile_name = self.tile_refs[index]
        if self._archive_path != tar_path:
            if self._archive is not None:
                self._archive.close()
            self._archive = tarfile.open(tar_path, "r")
            self._archive_path = tar_path

        member = self._archive.getmember(tile_name)
        file_obj = self._archive.extractfile(member)
        if file_obj is None:
            raise RuntimeError(f"Could not read {tile_name} from {tar_path}")
        image = Image.open(BytesIO(file_obj.read())).convert("RGB")
        return self.transform(image)


class BlockOutputCollector:
    """Capture transformer-block outputs from one forward pass."""

    def __init__(self, blocks):
        """Register one forward hook per transformer block."""
        self.outputs = [None] * len(blocks)
        self.handles = []
        for index, block in enumerate(blocks):
            self.handles.append(block.register_forward_hook(self._make_hook(index)))

    def _make_hook(self, index):
        """Create a hook for one block index."""
        def hook(_module, _inputs, output):
            """Store one detached block output on CPU."""
            if isinstance(output, tuple):
                output = output[0]
            self.outputs[index] = output.detach().cpu()
        return hook

    def pop(self):
        """Return outputs from the latest forward pass and clear the buffer."""
        if any(output is None for output in self.outputs):
            raise RuntimeError("At least one transformer block did not produce a captured output.")
        outputs = self.outputs
        self.outputs = [None] * len(outputs)
        return outputs

    def close(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def resolve_tar(tar_root, slide_id):
    """Resolve the tar archive corresponding to a PANDA slide identifier."""
    slide_id = Path(str(slide_id)).stem
    exact = tar_root / f"{slide_id}.tar"
    if exact.exists():
        return exact
    matches = sorted(tar_root.glob(f"{slide_id}*.tar"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No tar archive found for PANDA slide {slide_id}")


def sample_tile_refs(tar_root, train_csv, max_tiles, seed):
    """Uniformly reservoir-sample image members from PANDA training-slide archives."""
    frame = pd.read_csv(train_csv)
    if "FILENAME" not in frame.columns:
        raise ValueError("PANDA split CSV must contain a FILENAME column.")

    rng = random.Random(seed)
    reservoir = []
    seen = 0
    for slide_id in frame["FILENAME"].dropna().astype(str).unique():
        tar_path = resolve_tar(Path(tar_root), slide_id)
        with tarfile.open(tar_path, "r") as archive:
            for member in archive.getmembers():
                if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                ref = (tar_path, member.name)
                seen += 1
                if len(reservoir) < max_tiles:
                    reservoir.append(ref)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < max_tiles:
                        reservoir[replacement] = ref

    if not reservoir:
        raise RuntimeError("No image tiles were found in the PANDA training archives.")
    rng.shuffle(reservoir)
    return reservoir


def patch_start_index(sequence_length):
    """Return the first patch-token index for the supported DINO-style backbones."""
    return 5 if sequence_length >= 260 else 1


def compute_patch_similarity(model, transform, tile_refs, batch_size, num_workers, device):
    """Average cosine similarity between each pair of transformer blocks over patch tokens."""
    device = torch.device(device)
    model.to(device).eval()
    blocks = list(get_blocks(model))
    collector = BlockOutputCollector(blocks)
    dataset = RandomTilesDataset(tile_refs, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    similarity_sum = torch.zeros((len(blocks), len(blocks)), dtype=torch.float64)
    count = 0
    try:
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                model(images)
            outputs = collector.pop()
            tokens = torch.stack(outputs, dim=1).float()  # [B, L, N, D]
            if tokens.ndim != 4:
                raise ValueError(f"Expected token outputs with shape [B, L, N, D], got {tokens.shape}")

            start = patch_start_index(tokens.shape[2])
            patches = tokens[:, :, start:, :]
            if patches.shape[2] == 0:
                raise ValueError("No patch tokens remained after removing prefix/register tokens.")
            patches = F.normalize(patches, dim=-1)
            similarity_sum += torch.einsum("blnd,bmnd->lm", patches, patches).double()
            count += patches.shape[0] * patches.shape[2]
    finally:
        collector.close()

    return (similarity_sum / count).float().numpy()


def main():
    """Compute the patch-token block-similarity matrix used for histopathology phase discovery."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=["H0-mini", "hibou"], required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing prepared PANDA tar archives.")
    parser.add_argument("--train-csv", type=Path, required=True, help="PANDA training split CSV.")
    parser.add_argument("--h0-model", type=str, default=None)
    parser.add_argument("--hibou-model-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    if args.model_name == "H0-mini" and not args.h0_model:
        raise ValueError("--h0-model is required for H0-mini.")
    if args.model_name == "hibou" and not args.hibou_model_dir:
        raise ValueError("--hibou-model-dir is required for Hibou-B.")

    device = torch.device(args.device)
    models = get_models(device, h0_model_path=args.h0_model, hibou_model_dir=args.hibou_model_dir)
    model, transform, _ = models[args.model_name]
    tile_refs = sample_tile_refs(args.data_root, args.train_csv, args.num_samples, args.seed)
    matrix = compute_patch_similarity(
        model,
        transform,
        tile_refs,
        args.batch_size,
        args.num_workers,
        args.device,
    )

    out_prefix = args.out_prefix or Path(f"{args.model_name}_block_similarities/{args.model_name}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = out_prefix.with_name(out_prefix.name + "_avgcos_patches.npy")
    np.save(matrix_path, matrix)
    print(f"Saved {matrix_path}", flush=True)


if __name__ == "__main__":
    main()
