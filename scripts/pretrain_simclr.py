#!/usr/bin/env python3
"""
Self-Supervised Pre-Training (SimCLR) — MitraSETI

Trains the CNN+Transformer backbone on cached spectrograms using
contrastive learning (SimCLR: Chen et al., 2020).  No labels required —
the model learns to distinguish different signal morphologies by
maximizing agreement between augmented views of the same spectrogram.

This produces a pre-trained backbone that extracts meaningful 128-dim
feature vectors, significantly improving downstream classification
and OOD detection even without labeled training data.

Augmentations:
  - Random frequency shift (±10% of bandwidth)
  - Random time crop and resize
  - Gaussian noise injection
  - Random channel masking (simulated RFI)
  - Random brightness/contrast scaling

Output:
  mitraseti_artifacts/models/pretrained_simclr_backbone.pt
  mitraseti_artifacts/models/pretrain_log.json

Usage:
  python scripts/pretrain_simclr.py
  python scripts/pretrain_simclr.py --epochs 50 --batch-size 64
  python scripts/pretrain_simclr.py --lr 0.001 --temperature 0.07
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

if "HDF5_PLUGIN_PATH" not in os.environ:
    os.environ["HDF5_PLUGIN_PATH"] = ""

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DATA_DIR, MODELS_DIR

logger = logging.getLogger("mitraseti.pretrain")

CACHE_DIR = DATA_DIR / "spectrogram_cache"


# ── Augmentations ─────────────────────────────────────────────────────────────

class SpectrogramAugmentor:
    """Applies random augmentations to spectrogram tensors for SimCLR."""

    def __init__(self, freq_bins: int = 256, time_steps: int = 64):
        self.freq_bins = freq_bins
        self.time_steps = time_steps

    def __call__(self, x):
        """Apply a random composition of augmentations.

        Args:
            x: numpy array of shape (freq_bins, time_steps)

        Returns:
            Augmented copy.
        """
        x = x.copy().astype(np.float32)
        rng = np.random.default_rng()

        # 1. Random frequency shift (±10%)
        if rng.random() > 0.3:
            shift = rng.integers(-self.freq_bins // 10, self.freq_bins // 10 + 1)
            x = np.roll(x, shift, axis=0)

        # 2. Random time crop + resize
        if rng.random() > 0.4:
            crop_frac = rng.uniform(0.7, 1.0)
            crop_len = max(4, int(self.time_steps * crop_frac))
            start = rng.integers(0, max(1, self.time_steps - crop_len + 1))
            cropped = x[:, start:start + crop_len]
            from scipy.ndimage import zoom
            zoom_factor = self.time_steps / cropped.shape[1]
            x = zoom(cropped, (1.0, zoom_factor), order=1)

        # 3. Gaussian noise injection
        if rng.random() > 0.3:
            noise_level = rng.uniform(0.01, 0.15) * np.std(x)
            x = x + rng.standard_normal(x.shape).astype(np.float32) * noise_level

        # 4. Random channel masking (simulated RFI flagging)
        if rng.random() > 0.5:
            n_mask = rng.integers(1, max(2, self.freq_bins // 20))
            mask_chans = rng.integers(0, self.freq_bins, size=n_mask)
            x[mask_chans, :] = np.median(x)

        # 5. Brightness/contrast scaling
        if rng.random() > 0.3:
            brightness = rng.uniform(0.8, 1.2)
            contrast = rng.uniform(0.8, 1.3)
            mean_val = x.mean()
            x = (x - mean_val) * contrast + mean_val * brightness

        return x


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_cached_spectrograms(
    cache_dir: Path,
    max_samples: int = 10000,
    freq_bins: int = 256,
    time_steps: int = 64,
) -> np.ndarray:
    """Load cached spectrogram .npz files from the pipeline.

    Returns array of shape (N, freq_bins, time_steps).
    """
    npz_files = sorted(cache_dir.glob("spec_*.npz"))
    if not npz_files:
        logger.warning(f"No cached spectrograms in {cache_dir}")
        return np.array([])

    npz_files = npz_files[:max_samples]
    logger.info(f"Loading {len(npz_files)} cached spectrograms from {cache_dir}")

    spectrograms = []
    for fp in npz_files:
        try:
            data = np.load(fp)
            spec = data["spectrogram"].astype(np.float32)

            if spec.shape != (freq_bins, time_steps):
                from scipy.ndimage import zoom
                zf = (freq_bins / spec.shape[0], time_steps / spec.shape[1])
                spec = zoom(spec, zf, order=1)

            # Normalize to zero mean, unit variance
            mean, std = spec.mean(), spec.std()
            if std > 0:
                spec = (spec - mean) / std

            spectrograms.append(spec)
        except Exception:
            continue

    if not spectrograms:
        return np.array([])

    return np.stack(spectrograms)


def generate_synthetic_spectrograms(
    n_samples: int = 500,
    freq_bins: int = 256,
    time_steps: int = 64,
) -> np.ndarray:
    """Generate synthetic spectrograms when no cached data is available."""
    logger.info(f"Generating {n_samples} synthetic spectrograms for pre-training")
    rng = np.random.default_rng(42)
    spectrograms = []

    for _ in range(n_samples):
        spec = rng.standard_normal((freq_bins, time_steps)).astype(np.float32) * 0.3

        n_signals = rng.integers(0, 4)
        for _ in range(n_signals):
            sig_type = rng.choice(["drift", "stationary", "broadband", "pulsed"])
            snr = rng.uniform(5, 50)

            if sig_type == "drift":
                f0 = rng.integers(20, freq_bins - 20)
                drift = rng.uniform(-0.5, 0.5)
                for t in range(time_steps):
                    ch = int(f0 + drift * t)
                    if 0 <= ch < freq_bins:
                        spec[ch, t] += snr * 0.1
            elif sig_type == "stationary":
                f0 = rng.integers(10, freq_bins - 10)
                width = rng.integers(1, 4)
                for df in range(-width, width + 1):
                    if 0 <= f0 + df < freq_bins:
                        spec[f0 + df, :] += snr * 0.05
            elif sig_type == "broadband":
                t_start = rng.integers(0, time_steps - 5)
                t_end = min(t_start + rng.integers(3, 10), time_steps)
                spec[:, t_start:t_end] += rng.standard_normal(
                    (freq_bins, t_end - t_start)
                ).astype(np.float32) * snr * 0.02
            elif sig_type == "pulsed":
                f0 = rng.integers(20, freq_bins - 20)
                period = rng.integers(3, 15)
                for t in range(0, time_steps, period):
                    if t < time_steps:
                        spec[max(0, f0 - 2):min(freq_bins, f0 + 3), t] += snr * 0.08

        mean, std = spec.mean(), spec.std()
        if std > 0:
            spec = (spec - mean) / std
        spectrograms.append(spec)

    return np.stack(spectrograms)


# ── SimCLR Training ───────────────────────────────────────────────────────────

def nt_xent_loss(z_i, z_j, temperature=0.07):
    """Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    The core SimCLR contrastive loss that pushes together representations
    of augmented views of the same image and apart representations of
    different images.
    """
    import torch
    import torch.nn.functional as F

    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)

    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+N) and (i+N, i)
    pos_i = torch.arange(batch_size, device=z.device)
    pos_j = pos_i + batch_size

    labels_top = pos_j
    labels_bottom = pos_i
    labels = torch.cat([labels_top, labels_bottom], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss


def train_simclr(
    spectrograms: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    temperature: float = 0.07,
    embed_dim: int = 128,
    freq_bins: int = 256,
    time_steps: int = 64,
) -> dict:
    """Train the backbone with SimCLR contrastive learning."""
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference.signal_classifier import _build_model

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Training on: {device}")

    model = _build_model(num_classes=9, freq_bins=freq_bins, time_steps=time_steps)

    # Projection head for contrastive learning
    projection_head = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.GELU(),
        nn.Linear(embed_dim, 64),
    )

    model.to(device)
    projection_head.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    augmentor = SpectrogramAugmentor(freq_bins, time_steps)

    n_samples = len(spectrograms)
    n_batches = max(1, n_samples // batch_size)

    log_entries = []
    best_loss = float("inf")

    logger.info(f"SimCLR training: {epochs} epochs, {n_samples} samples, batch={batch_size}")

    for epoch in range(epochs):
        model.train()
        projection_head.train()
        epoch_loss = 0.0
        n_steps = 0

        indices = np.random.permutation(n_samples)

        for b in range(n_batches):
            batch_idx = indices[b * batch_size:(b + 1) * batch_size]
            if len(batch_idx) < 2:
                continue

            batch_specs = spectrograms[batch_idx]

            # Generate two augmented views of each spectrogram
            views_i = np.stack([augmentor(s) for s in batch_specs])
            views_j = np.stack([augmentor(s) for s in batch_specs])

            x_i = torch.from_numpy(views_i).float().to(device)
            x_j = torch.from_numpy(views_j).float().to(device)

            _, feat_i = model(x_i, return_features=True)
            _, feat_j = model(x_j, return_features=True)

            z_i = projection_head(feat_i)
            z_j = projection_head(feat_j)

            loss = nt_xent_loss(z_i, z_j, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        entry = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "lr": round(current_lr, 8),
        }
        log_entries.append(entry)

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best backbone weights (without projection head)
            backbone_path = MODELS_DIR / "pretrained_simclr_backbone.pt"
            torch.save(model.state_dict(), backbone_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  lr={current_lr:.6f}  "
                f"{'*best*' if avg_loss <= best_loss else ''}"
            )

    return {
        "epochs": epochs,
        "best_loss": round(best_loss, 6),
        "final_loss": round(avg_loss, 6),
        "n_samples": n_samples,
        "batch_size": batch_size,
        "temperature": temperature,
        "device": device,
        "log": log_entries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MitraSETI Self-Supervised Pre-Training (SimCLR)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="NT-Xent temperature (default: 0.07)")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max spectrograms to load (default: 5000)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data even if cache exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load data
    spectrograms = np.array([])
    if not args.synthetic and CACHE_DIR.exists():
        spectrograms = load_cached_spectrograms(
            CACHE_DIR, max_samples=args.max_samples
        )

    if len(spectrograms) < 50:
        logger.info("Insufficient cached data, generating synthetic spectrograms")
        synthetic = generate_synthetic_spectrograms(n_samples=max(500, args.max_samples))
        if len(spectrograms) > 0:
            spectrograms = np.concatenate([spectrograms, synthetic])
        else:
            spectrograms = synthetic

    logger.info(f"Training dataset: {len(spectrograms)} spectrograms")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = train_simclr(
        spectrograms,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
    )

    results["timestamp"] = datetime.now().isoformat()
    log_path = MODELS_DIR / "pretrain_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    backbone_path = MODELS_DIR / "pretrained_simclr_backbone.pt"
    print(f"\n{'='*60}")
    print("SELF-SUPERVISED PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Epochs:       {results['epochs']}")
    print(f"  Best loss:    {results['best_loss']:.4f}")
    print(f"  Final loss:   {results['final_loss']:.4f}")
    print(f"  Samples:      {results['n_samples']}")
    print(f"  Device:       {results['device']}")
    print(f"  Backbone:     {backbone_path}")
    print(f"  Training log: {log_path}")
    print("\nTo use the pre-trained backbone:")
    print(f"  pipe = MitraSETIPipeline(model_path='{backbone_path}')")


if __name__ == "__main__":
    main()
