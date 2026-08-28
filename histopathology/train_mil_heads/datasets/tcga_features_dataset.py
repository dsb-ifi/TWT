from pathlib import Path

import torch
from torch.utils.data import Dataset


class TCGAFeaturesDataset(Dataset):
    """Load one frozen tile-feature bag per TCGA-PRAD scan."""

    def __init__(self, feature_dir):
        """Index every per-scan .pt feature file in a directory."""
        self.feature_paths = sorted(Path(feature_dir).glob("*.pt"))
        if not self.feature_paths:
            raise FileNotFoundError(f"No .pt feature files found in {feature_dir}")

    def __len__(self):
        """Return the number of indexed TCGA-PRAD scans."""
        return len(self.feature_paths)

    def __getitem__(self, index):
        """Return all frozen tile features for one TCGA-PRAD scan."""
        feature_path = self.feature_paths[index]
        obj = torch.load(feature_path, map_location="cpu")
        features = obj["features"].float()
        if features.ndim != 2:
            raise ValueError(f"Expected [N, D] features in {feature_path}, got {features.shape}")
        return {"features": features, "filename": feature_path.stem}


def tcga_collate_fn(batch):
    """Collate feature bags; the paper experiments use batch size one scan."""
    if len(batch) != 1:
        raise ValueError("TCGA-PRAD evaluation uses batch_size=1 for variable-size WSI bags.")
    item = batch[0]
    return {"features": item["features"].unsqueeze(0), "filename": [item["filename"]]}
