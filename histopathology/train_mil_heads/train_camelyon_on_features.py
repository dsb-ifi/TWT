import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from histopathology.train_mil_heads.datasets.camelyon_mil_dataset import CamelyonFeaturesDataset, camelyon_collate_fn
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


def build_loader(split_csv, split, feature_dir, shuffle, num_workers):
    """Build a feature-only CAMELYON loader for one named split."""
    dataset = CamelyonFeaturesDataset(split_csv, split, feature_dir)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=camelyon_collate_fn,
    )


def make_loaders(args):
    """Build CAMELYON17 train/validation/test and CAMELYON16 external-test loaders."""
    return (
        build_loader(args.camelyon17_splits_csv, "train", args.camelyon17_features, True, args.num_workers),
        build_loader(args.camelyon17_splits_csv, "val", args.camelyon17_features, False, args.num_workers),
        build_loader(args.camelyon17_splits_csv, "test", args.camelyon17_features, False, args.num_workers),
        build_loader(args.camelyon16_splits_csv, "test", args.camelyon16_features, False, args.num_workers),
    )


def build_mil_model(method, feature_size):
    """Construct the ABMIL or TransMIL head used in the paper."""
    if method == "abmil":
        return ABMIL(in_dim=feature_size, hidden_dim=128, num_classes=2)
    if method == "transmil":
        return TransMIL(in_dim=feature_size, hidden_dim=128, num_classes=2)
    raise ValueError(f"Unsupported MIL method: {method}")


def run_epoch(dataloader, epoch, loss_fn, model, optimizer, accumulation_steps, device):
    """Train one CAMELYON17 epoch with gradient accumulation and norm clipping."""
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
def evaluate(model, dataloader, criterion, device):
    """Return balanced accuracy, accuracy, ROC AUC, and loss for a CAMELYON split."""
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_score = [], [], []
    for batch in dataloader:
        labels = batch["label"].to(device, non_blocking=True)
        logits, _ = model(batch["features"].to(device, non_blocking=True))
        probabilities = torch.softmax(logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy())
        y_score.extend(probabilities[:, 1].cpu().numpy())
        total_loss += criterion(logits, labels).item()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    return (
        float(balanced_accuracy_score(y_true, y_pred)),
        float(accuracy_score(y_true, y_pred)),
        float(roc_auc_score(y_true, y_score)),
        total_loss / max(len(dataloader), 1),
    )


def moving_average(values, window=5):
    """Return the trailing moving average used for CAMELYON17 checkpoint selection."""
    return float(np.mean(values[-window:]))


def train(args):
    """Train a MIL head and select results by CAMELYON17 validation-accuracy moving average."""
    device = torch.device(args.device)
    feature_size = args.feature_size or infer_feature_size(args.camelyon17_features)
    model = build_mil_model(args.method, feature_size).to(device)
    train_loader, val_loader, test_loader, external_loader = make_loaders(args)

    class_counts = torch.tensor(
        train_loader.dataset.df["label"].value_counts().sort_index().values,
        dtype=torch.float,
        device=device,
    )
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()
    train_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    eval_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    val_history = []
    rows = []
    best_score = float("-inf")
    best_row = None

    for epoch in range(args.epochs):
        start = time.time()
        run_epoch(train_loader, epoch, train_criterion, model, optimizer, args.grad_accumulation_count, device)
        scheduler.step()

        _, val_acc, val_auc, val_loss = evaluate(model, val_loader, eval_criterion, device)
        _, test_acc, test_auc, _ = evaluate(model, test_loader, eval_criterion, device)
        _, ext_acc, ext_auc, _ = evaluate(model, external_loader, eval_criterion, device)

        val_history.append(val_acc)
        score = moving_average(val_history)
        row = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_acc_ma5": score,
            "cam17_test_acc": test_acc,
            "cam17_test_auc": test_auc,
            "cam16_acc": ext_acc,
            "cam16_auc": ext_auc,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_row = row.copy()

        print(
            f"Epoch {epoch:03d}: val acc={100 * val_acc:.2f}, ma5={100 * score:.2f}, "
            f"CAM17 test={100 * test_acc:.2f}, CAM16={100 * ext_acc:.2f}; "
            f"time={timedelta(seconds=time.time() - start)}",
            flush=True,
        )
        if args.metrics_csv:
            args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.metrics_csv, index=False)

    if best_row is None:
        raise RuntimeError("Training produced no evaluation rows.")
    print(
        "Selected by highest 5-epoch moving average of CAMELYON17 validation accuracy: "
        f"epoch {best_row['epoch']}; CAM17 test={100 * best_row['cam17_test_acc']:.2f}; "
        f"CAM16={100 * best_row['cam16_acc']:.2f}",
        flush=True,
    )


def get_args():
    """Parse paths and hyperparameters for the CAMELYON17/CAMELYON16 MIL experiment."""
    parser = argparse.ArgumentParser(description="Train ABMIL or TransMIL on frozen CAMELYON17 tile features.")
    parser.add_argument("--method", choices=["abmil", "transmil"], default="abmil")
    parser.add_argument("--camelyon17-features", type=Path, required=True)
    parser.add_argument("--camelyon17-splits-csv", type=Path, required=True)
    parser.add_argument("--camelyon16-features", type=Path, required=True)
    parser.add_argument("--camelyon16-splits-csv", type=Path, required=True)
    parser.add_argument("--feature-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad-accumulation-count", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=min((os.cpu_count() or 2) // 2, 32))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics-csv", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
