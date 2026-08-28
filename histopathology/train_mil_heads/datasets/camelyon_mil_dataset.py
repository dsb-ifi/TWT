from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class CamelyonFeaturesDataset(Dataset):
    """Load frozen per-slide tile-feature bags for a named CAMELYON split."""

    def __init__(self, split_csv, split, feature_dir):
        """Select train, validation, or test slides and index their feature files."""
        self.feature_dir = Path(feature_dir)
        frame = pd.read_csv(split_csv, dtype={"slide_id": str, "split": str})
        required = {"slide_id", "label", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CAMELYON split CSV is missing columns: {sorted(missing)}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        self.df = frame[frame["split"] == split].reset_index(drop=True)

    def __len__(self):
        """Return the number of slides in the selected split."""
        return len(self.df)

    def __getitem__(self, index):
        """Return all frozen tile features and the binary slide label."""
        row = self.df.iloc[index]
        slide_id = Path(str(row["slide_id"])).stem
        feature_path = self.feature_dir / f"{slide_id}.pt"
        if not feature_path.exists():
            raise FileNotFoundError(f"No feature file found for CAMELYON slide {slide_id}")
        obj = torch.load(feature_path, map_location="cpu")
        features = obj["features"].float()
        if features.ndim != 2:
            raise ValueError(f"Expected [N, D] features in {feature_path}, got {features.shape}")
        return {
            "features": features,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "slide_id": slide_id,
        }


def camelyon_collate_fn(batch):
    """Collate CAMELYON feature bags; the paper experiments use batch size one WSI."""
    if len(batch) != 1:
        raise ValueError("CAMELYON MIL uses batch_size=1 because WSIs contain different numbers of tiles.")
    item = batch[0]
    return {
        "features": item["features"].unsqueeze(0),
        "label": item["label"].unsqueeze(0),
        "slide_id": [item["slide_id"]],
    }
