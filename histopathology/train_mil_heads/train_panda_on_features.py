import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from histopathology.train_mil_heads.datasets.panda_features_dataset import PandaFeaturesDataset, panda_collate_fn
from histopathology.train_mil_heads.datasets.tcga_features_dataset import TCGAFeaturesDataset, tcga_collate_fn
from histopathology.train_mil_heads.mil.abmil import ABMIL
from histopathology.train_mil_heads.mil.transmil import TransMIL


def infer_feature_size(feature_dir):
    """Infer embedding width from the first saved per-slide feature file."""
    feature_files = sorted(Path(feature_dir).glob("*.pt"))
    if not feature_files:
        raise FileNotFoundError(f"No .pt feature files found in {feature_dir}")
    features = torch.load(feature_files[0], map_location="cpu")["features"]
    if features.ndim != 2:
        raise ValueError(f"Expected [N, D] features in {feature_files[0]}, got {features.shape}")
    return int(features.shape[1])


def make_loaders(args):
    """Build PANDA train/validation/test loaders and the TCGA-PRAD external-test loader."""
    train_ds = PandaFeaturesDataset(args.train_csv, args.panda_features)
    val_ds = PandaFeaturesDataset(args.val_csv, args.panda_features)
    test_ds = PandaFeaturesDataset(args.test_csv, args.panda_features)
    tcga_ds = TCGAFeaturesDataset(args.tcga_features)

    loader_kwargs = {"batch_size": 1, "num_workers": args.num_workers, "pin_memory": True}
    return (
        DataLoader(train_ds, shuffle=True, collate_fn=panda_collate_fn, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, collate_fn=panda_collate_fn, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, collate_fn=panda_collate_fn, **loader_kwargs),
        DataLoader(tcga_ds, shuffle=False, collate_fn=tcga_collate_fn, **loader_kwargs),
    )


def build_mil_model(method, feature_size):
    """Construct the ABMIL or TransMIL head used in the paper."""
    if method == "abmil":
        return ABMIL(in_dim=feature_size, hidden_dim=128, num_classes=6)
    if method == "transmil":
        return TransMIL(in_dim=feature_size, hidden_dim=128, num_classes=6)
    raise ValueError(f"Unsupported MIL method: {method}")


def run_epoch(dataloader, epoch, loss_fn, model, optimizer, accumulation_steps, device):
    """Train one PANDA epoch with gradient accumulation and norm clipping."""
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for step, batch in enumerate(dataloader, 1):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, _ = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()

        if step % accumulation_steps == 0 or step == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch:03d} train loss: {total_loss / max(len(dataloader), 1):.4f}", flush=True)


@torch.no_grad()
def evaluate_panda(model, dataloader, device):
    """Return quadratic weighted kappa for a PANDA split."""
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        logits, _ = model(batch["features"].to(device, non_blocking=True))
        y_pred.extend(logits.argmax(dim=1).cpu().numpy())
        y_true.extend(batch["label"].numpy())
    return float(cohen_kappa_score(np.asarray(y_true), np.asarray(y_pred), weights="quadratic"))


@torch.no_grad()
def evaluate_tcga(model, dataloader, device, ground_truth):
    """Return TCGA-PRAD quadratic weighted kappa for scans with clinical labels."""
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        logits, _ = model(batch["features"].to(device, non_blocking=True))
        predictions = logits.argmax(dim=1).cpu().numpy()
        for filename, prediction in zip(batch["filename"], predictions):
            if filename in ground_truth:
                y_true.append(ground_truth[filename])
                y_pred.append(int(prediction))
    if not y_true:
        raise RuntimeError("No TCGA feature filenames matched the clinical metadata.")
    return float(cohen_kappa_score(np.asarray(y_true), np.asarray(y_pred), weights="quadratic"))


def gleason_to_isup(primary, secondary):
    """Convert primary and secondary Gleason patterns to ISUP grade groups."""
    primary = int(str(primary).split()[-1])
    secondary = int(str(secondary).split()[-1])
    if primary + secondary == 6:
        return 1
    if primary == 3 and secondary == 4:
        return 2
    if primary == 4 and secondary == 3:
        return 3
    if primary + secondary == 8:
        return 4
    if primary + secondary in (9, 10):
        return 5
    return 0


def load_tcga_ground_truth(clinical_csv):
    """Load TCGA-PRAD scan identifiers and derive their ISUP grade groups."""
    frame = pd.read_csv(clinical_csv)
    required = {"scan_name_aperio", "gleason_1", "gleason_2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"TCGA clinical CSV is missing columns: {sorted(missing)}")
    frame["isup_grade"] = frame.apply(
        lambda row: gleason_to_isup(row["gleason_1"], row["gleason_2"]), axis=1
    )
    frame["scan_name_aperio"] = frame["scan_name_aperio"].str.replace(".svs", "", regex=False)
    return dict(zip(frame["scan_name_aperio"], frame["isup_grade"]))


def train(args):
    """Train a MIL head and select the epoch with the highest PANDA validation QWK."""
    device = torch.device(args.device)
    feature_size = args.feature_size or infer_feature_size(args.panda_features)
    model = build_mil_model(args.method, feature_size).to(device)
    train_loader, val_loader, test_loader, tcga_loader = make_loaders(args)
    tcga_ground_truth = load_tcga_ground_truth(args.tcga_clinical_csv)

    class_counts = torch.tensor(
        train_loader.dataset.df["isup_grade"].value_counts().sort_index().values,
        dtype=torch.float,
        device=device,
    )
    class_weights = 1.0 / (class_counts + 1e-12)
    class_weights /= class_weights.sum()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = {"val": float("-inf"), "epoch": -1, "test": float("nan"), "tcga": float("nan")}
    for epoch in range(args.epochs):
        start = time.time()
        run_epoch(train_loader, epoch, loss_fn, model, optimizer, args.grad_accumulation_count, device)
        scheduler.step()
        val_qwk = evaluate_panda(model, val_loader, device)
        print(f"Epoch {epoch:03d} val QWK: {100 * val_qwk:.4f}", flush=True)

        if val_qwk > best["val"]:
            best = {
                "val": val_qwk,
                "epoch": epoch,
                "test": evaluate_panda(model, test_loader, device),
                "tcga": evaluate_tcga(model, tcga_loader, device, tcga_ground_truth),
            }
            print(
                f"  new best -> PANDA test {100 * best['test']:.4f}, "
                f"TCGA-PRAD {100 * best['tcga']:.4f}",
                flush=True,
            )
        print(f"Epoch time: {timedelta(seconds=time.time() - start)}", flush=True)

    print(
        f"Best validation epoch: {best['epoch']}; PANDA test QWK={100 * best['test']:.4f}; "
        f"TCGA-PRAD QWK={100 * best['tcga']:.4f}",
        flush=True,
    )


def get_args():
    """Parse paths and hyperparameters for the PANDA/TCGA-PRAD MIL experiment."""
    parser = argparse.ArgumentParser(description="Train ABMIL or TransMIL on frozen PANDA tile features.")
    parser.add_argument("--method", choices=["abmil", "transmil"], default="abmil")
    parser.add_argument("--panda-features", type=Path, required=True)
    parser.add_argument("--tcga-features", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--tcga-clinical-csv", type=Path, required=True)
    parser.add_argument("--feature-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--grad-accumulation-count", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=min((os.cpu_count() or 2) // 2, 12))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
