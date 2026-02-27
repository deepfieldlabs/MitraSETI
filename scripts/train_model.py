#!/usr/bin/env python3
"""
Model Training Script for MitraSETI Signal Classifier

Trains the CNN + Transformer hybrid model on synthetic spectrograms.
After training, extracts penultimate-layer embeddings for OOD calibration
(per-class Mahalanobis distance).

Usage:
    python scripts/train_model.py --data-dir data/training --epochs 50 --batch-size 64

Prerequisites:
    Run scripts/generate_training_data.py first to create training data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from inference.signal_classifier import _build_model, SIGNAL_LABELS

logger = logging.getLogger(__name__)

# Signal class names (must match generate_training_data.py ordering)
CLASS_NAMES = list(SIGNAL_LABELS)
N_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select the best available training device.
    
    Note: MPS has known numerical issues with some operations.
    Set MITRASETI_DEVICE=cpu to force CPU.
    """
    import os
    forced = os.environ.get("MITRASETI_DEVICE", "").lower()
    if forced == "cpu":
        logger.info("Using CPU (forced via MITRASETI_DEVICE)")
        return torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        dev = torch.device("cpu")
        logger.info("Using CPU")
    return dev


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    data_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray]:
    """Load spectrograms and labels, split into train/val.

    Returns:
        train_dataset, val_dataset, full spectrograms array, full labels array
    """
    spec_path = data_dir / "spectrograms.npy"
    labels_path = data_dir / "labels.npy"

    if not spec_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_dir}. "
            "Run scripts/generate_training_data.py first."
        )

    specs = np.load(spec_path)
    labels = np.load(labels_path)

    logger.info(f"Loaded {specs.shape[0]} spectrograms, shape={specs.shape[1:]}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(specs))
    split = int(len(indices) * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    train_specs = torch.from_numpy(specs[train_idx]).float()
    train_labels = torch.from_numpy(labels[train_idx]).long()
    val_specs = torch.from_numpy(specs[val_idx]).float()
    val_labels = torch.from_numpy(labels[val_idx]).long()

    for i in range(len(train_specs)):
        s = train_specs[i]
        m, st = s.mean(), s.std()
        if st > 1e-8:
            train_specs[i] = (s - m) / st
    for i in range(len(val_specs)):
        s = val_specs[i]
        m, st = s.mean(), s.std()
        if st > 1e-8:
            val_specs[i] = (s - m) / st

    train_ds = TensorDataset(train_specs, train_labels)
    val_ds = TensorDataset(val_specs, val_labels)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds, specs, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_specs, batch_labels in loader:
        batch_specs = batch_specs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(batch_specs)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_specs)
            loss = criterion(logits, batch_labels)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * batch_specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_labels).sum().item()
        total += batch_specs.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_specs, batch_labels in loader:
        batch_specs = batch_specs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        logits = model(batch_specs)
        loss = criterion(logits, batch_labels)
        if torch.isnan(loss):
            continue
        total_loss += loss.item() * batch_specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_labels).sum().item()
        total += batch_specs.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_class_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 per class."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch_specs, batch_labels in loader:
        batch_specs = batch_specs.to(device, non_blocking=True)
        logits = model(batch_specs)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(batch_labels)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics = {}
    for cls in range(n_classes):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        metrics[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int((labels == cls).sum()),
        }

    return metrics


# ---------------------------------------------------------------------------
# Embedding extraction and OOD calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    specs: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Extract penultimate-layer embeddings for all training data.

    The model's forward(return_features=True) returns (logits, features).
    """
    model.eval()
    all_embeddings = []
    n = len(specs)

    mean = specs.mean()
    std = specs.std()
    if std > 0:
        specs_norm = (specs - mean) / std
    else:
        specs_norm = specs - mean

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(specs_norm[i : i + batch_size]).float().to(device)
        _, features = model(batch, return_features=True)
        all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def compute_ood_calibration(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> dict:
    """Compute per-class mean and covariance for Mahalanobis OOD detection.

    Returns a JSON-serializable calibration dict.
    """
    calibration = {
        "n_classes": n_classes,
        "embed_dim": int(embeddings.shape[1]),
        "class_stats": {},
    }

    for cls in range(n_classes):
        mask = labels == cls
        if mask.sum() < 2:
            continue
        cls_emb = embeddings[mask]
        mean = cls_emb.mean(axis=0)
        cov = np.cov(cls_emb, rowvar=False)
        # Regularize for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        calibration["class_stats"][name] = {
            "mean": mean.tolist(),
            "covariance": cov.tolist(),
            "n_samples": int(mask.sum()),
        }

    # Global statistics for spectral threshold
    global_mean = embeddings.mean(axis=0)
    global_cov = np.cov(embeddings, rowvar=False)
    global_cov += np.eye(global_cov.shape[0]) * 1e-6

    distances = []
    inv_cov = np.linalg.inv(global_cov)
    for i in range(len(embeddings)):
        diff = embeddings[i] - global_mean
        d = float(np.sqrt(diff @ inv_cov @ diff))
        distances.append(d)

    calibration["global_mean"] = global_mean.tolist()
    calibration["spectral_threshold"] = float(np.percentile(distances, 95))
    calibration["distance_percentiles"] = {
        "p50": float(np.percentile(distances, 50)),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
    }

    return calibration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MitraSETI signal classifier model."
    )
    parser.add_argument("--data-dir", type=str, default="data/training",
                        help="Directory containing spectrograms.npy and labels.npy")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory for model weights and calibration output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = _PROJECT_ROOT
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Load data
    train_ds, val_ds, full_specs, full_labels = load_data(data_dir, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Build model (uses _build_model factory which creates SignalClassifierModel
    # with CNN backbone + Transformer encoder + MLP head)
    n_freq = full_specs.shape[1]
    n_time = full_specs.shape[2]
    model = _build_model(num_classes=N_CLASSES, freq_bins=n_freq, time_steps=n_time)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,} total, {trainable:,} trainable")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Training
    model_path = output_dir / "signal_classifier_v1.pt"
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    print(f"\nMitraSETI Model Training")
    print(f"  Device:     {device}")
    print(f"  Model:      CNN + Transformer ({param_count:,} params)")
    print(f"  Data:       {len(train_ds)} train / {len(val_ds)} val")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  AMP:        {use_amp}")
    print(f"  Output:     {model_path}")
    print()

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        ep_time = time.time() - t_ep

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}  "
            f"{'*BEST*' if epoch == best_epoch and epoch == len(history['val_acc']) else ''}  "
            f"({ep_time:.1f}s)"
        )

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # Reload best weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Per-class metrics on validation set
    print(f"\nPer-class metrics (validation):")
    print(f"{'Class':25s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    print("-" * 60)
    per_class = compute_per_class_metrics(model, val_loader, device, N_CLASSES)
    for name, m in per_class.items():
        print(f"{name:25s} {m['precision']:8.4f} {m['recall']:8.4f} {m['f1']:8.4f} {m['support']:8d}")

    # Extract embeddings for OOD calibration
    print(f"\nExtracting embeddings for OOD calibration...")
    embeddings = extract_embeddings(model, full_specs, full_labels, device, batch_size=args.batch_size)
    emb_path = output_dir / "training_embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info(f"Embeddings saved: {embeddings.shape} → {emb_path}")

    print(f"Computing OOD calibration (per-class Mahalanobis)...")
    calibration = compute_ood_calibration(embeddings, full_labels, N_CLASSES)
    calibration["training_info"] = {
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "total_time_s": round(total_time, 1),
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    calibration["per_class_metrics"] = per_class

    cal_path = output_dir / "ood_calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"OOD calibration saved → {cal_path}")

    # Save training history
    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nOutputs:")
    print(f"  Model weights:     {model_path}")
    print(f"  Embeddings:        {emb_path}")
    print(f"  OOD calibration:   {cal_path}")
    print(f"  Training history:  {hist_path}")
    print(f"\nFinal val accuracy:  {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
