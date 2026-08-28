from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class PandaFeaturesDataset(Dataset):
    """Load one frozen tile-feature bag per PANDA WSI."""

    def __init__(self, csv_path, feature_dir):
        """Index slide identifiers and ISUP labels from a PANDA split CSV."""
        self.feature_dir = Path(feature_dir)
        self.df = pd.read_csv(csv_path)
        required = {"FILENAME", "isup_grade"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"PANDA CSV is missing columns: {sorted(missing)}")
        self.df = self.df.drop_duplicates("FILENAME").reset_index(drop=True)

    def __len__(self):
        """Return the number of PANDA WSIs in the split."""
        return len(self.df)

    def __getitem__(self, index):
        """Return all frozen tile features and the WSI-level ISUP label."""
        row = self.df.iloc[index]
        slide_id = Path(str(row["FILENAME"])).stem
        feature_path = self.feature_dir / f"{slide_id}.pt"
        if not feature_path.exists():
            matches = sorted(self.feature_dir.glob(f"{slide_id}*.pt"))
            if not matches:
                raise FileNotFoundError(f"No feature file found for PANDA slide {slide_id}")
            feature_path = matches[0]

        obj = torch.load(feature_path, map_location="cpu")
        features = obj["features"].float()
        if features.ndim != 2:
            raise ValueError(f"Expected [N, D] features in {feature_path}, got {features.shape}")
        return {
            "features": features,
            "label": torch.tensor(int(row["isup_grade"]), dtype=torch.long),
            "filename": slide_id,
        }


def panda_collate_fn(batch):
    """Collate feature bags; the paper experiments use batch size one WSI."""
    if len(batch) != 1:
        raise ValueError("PANDA MIL uses batch_size=1 because WSIs contain different numbers of tiles.")
    item = batch[0]
    return {
        "features": item["features"].unsqueeze(0),
        "label": item["label"].unsqueeze(0),
        "filename": [item["filename"]],
    }
