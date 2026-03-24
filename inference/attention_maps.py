"""
Transformer Attention Heatmap Extraction — MitraSETI

Extracts and visualizes self-attention weights from the Transformer
encoder layers.  Shows which time-frequency regions the model attends
to when making classification decisions — critical for interpretability
and publication.

Usage:
    from inference.attention_maps import extract_attention, plot_attention_overlay

    attn_maps = extract_attention(model, spectrogram, device)
    plot_attention_overlay(spectrogram, attn_maps, output_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_attention(
    model,
    spectrogram: np.ndarray,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Extract attention weights from all Transformer encoder layers.

    Registers forward hooks on the MultiheadAttention modules inside the
    TransformerEncoder to capture the attention weight matrices during
    a forward pass.

    Args:
        model: A SignalClassifierModel instance.
        spectrogram: 2D array (freq_bins, time_steps).
        device: Torch device string.

    Returns:
        Dict with:
          - "layer_0", "layer_1", ...: attention weights (n_heads, T, T)
          - "mean": averaged across layers and heads (T, T)
          - "cnn_features": per-timestep CNN embeddings (T, embed_dim)
    """
    import torch

    hooks = []
    attention_weights = {}

    def _make_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn = output[1]
                if attn is not None:
                    attention_weights[layer_name] = attn.detach().cpu().numpy()
        return hook_fn

    # Register hooks on MultiheadAttention self_attn in each encoder layer
    for i, layer in enumerate(model.transformer.layers):
        h = layer.self_attn.register_forward_hook(_make_hook(f"layer_{i}"))
        hooks.append(h)

    # Preprocess spectrogram
    from scipy.ndimage import zoom

    spec = spectrogram.astype(np.float32)
    if spec.shape != (model.freq_bins, model.time_steps):
        zf = (model.freq_bins / spec.shape[0], model.time_steps / spec.shape[1])
        spec = zoom(spec, zf, order=1)

    mean, std = spec.mean(), spec.std()
    if std > 0:
        spec = (spec - mean) / std

    tensor = torch.from_numpy(spec).unsqueeze(0).to(device)

    # Forward pass with attention capture
    model.eval()
    with torch.no_grad():
        # Temporarily enable need_weights on all self_attn layers
        original_flags = []
        for layer in model.transformer.layers:
            original_flags.append(getattr(layer.self_attn, '_qkv_same_embed_dim', True))

        x = tensor.transpose(1, 2)  # (1, T, F)
        cnn_out = model.cnn_backbone(x)  # (1, T, embed_dim)
        cnn_features = cnn_out[0].detach().cpu().numpy()

        x_enc = model.pos_encoder(cnn_out)
        x_enc = model.dropout(x_enc)

        # Run through encoder layers manually to get attention
        out = x_enc
        for i, layer in enumerate(model.transformer.layers):
            # Self-attention with weights
            attn_out, attn_w = layer.self_attn(
                out, out, out, need_weights=True, average_attn_weights=False
            )
            out = layer.norm1(out + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(out))))
            out = layer.norm2(out + layer.dropout2(ff_out))

            if attn_w is not None:
                attention_weights[f"layer_{i}"] = attn_w[0].detach().cpu().numpy()

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Compute mean attention across layers and heads
    all_attn = []
    for key in sorted(attention_weights.keys()):
        w = attention_weights[key]
        if w.ndim == 3:  # (n_heads, T, T)
            all_attn.append(w.mean(axis=0))
        elif w.ndim == 2:  # (T, T)
            all_attn.append(w)

    if all_attn:
        attention_weights["mean"] = np.mean(all_attn, axis=0)
    else:
        T = model.time_steps
        attention_weights["mean"] = np.ones((T, T)) / T

    attention_weights["cnn_features"] = cnn_features

    return attention_weights


def attention_to_spectrogram_heatmap(
    attn_mean: np.ndarray,
    spectrogram: np.ndarray,
) -> np.ndarray:
    """Convert temporal attention to a time-frequency heatmap.

    The attention matrix is (T, T) — it shows how each time step attends
    to every other.  We collapse this to per-timestep importance (sum of
    incoming attention), then tile across frequency to create a heatmap
    matching the spectrogram dimensions.

    Returns:
        Heatmap array of same shape as spectrogram, values in [0, 1].
    """
    # Per-timestep importance: how much total attention each step receives
    importance = attn_mean.sum(axis=0)  # (T,)
    importance = importance / importance.max() if importance.max() > 0 else importance

    # Tile across frequency dimension
    n_freq, n_time = spectrogram.shape

    if len(importance) != n_time:
        from scipy.ndimage import zoom
        importance = zoom(importance, n_time / len(importance), order=1)

    heatmap = np.tile(importance, (n_freq, 1))
    return heatmap


def plot_attention_overlay(
    spectrogram: np.ndarray,
    attention_weights: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "",
) -> None:
    """Generate a publication-ready attention overlay figure.

    Creates a 2x2 grid:
      - Top-left: original spectrogram
      - Top-right: attention heatmap overlaid on spectrogram
      - Bottom-left: per-head attention matrices (if multi-head)
      - Bottom-right: temporal importance profile
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#080c14")

    attn_mean = attention_weights.get("mean", np.ones((64, 64)) / 64)
    heatmap = attention_to_spectrogram_heatmap(attn_mean, spectrogram)

    # Top-left: raw spectrogram
    ax = axes[0, 0]
    ax.set_facecolor("#080c14")
    ax.imshow(spectrogram, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("Input Spectrogram", color="#e0e8f0", fontsize=12, fontweight=300)
    ax.set_ylabel("Frequency", color="#8ca5c8")
    ax.tick_params(colors="#8ca5c8")

    # Top-right: attention overlay
    ax = axes[0, 1]
    ax.set_facecolor("#080c14")
    ax.imshow(spectrogram, aspect="auto", cmap="viridis", alpha=0.6, interpolation="nearest")
    ax.imshow(heatmap, aspect="auto", cmap="hot", alpha=0.4, interpolation="nearest")
    ax.set_title("Attention Overlay", color="#e0e8f0", fontsize=12, fontweight=300)
    ax.tick_params(colors="#8ca5c8")

    # Bottom-left: attention matrix
    ax = axes[1, 0]
    ax.set_facecolor("#080c14")
    ax.imshow(attn_mean, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_title("Attention Matrix (mean)", color="#e0e8f0", fontsize=12, fontweight=300)
    ax.set_xlabel("Key Time Step", color="#8ca5c8")
    ax.set_ylabel("Query Time Step", color="#8ca5c8")
    ax.tick_params(colors="#8ca5c8")

    # Bottom-right: temporal importance
    ax = axes[1, 1]
    ax.set_facecolor("#080c14")
    importance = attn_mean.sum(axis=0)
    importance = importance / importance.max() if importance.max() > 0 else importance
    ax.fill_between(range(len(importance)), importance, alpha=0.3, color="#00d4ff")
    ax.plot(importance, color="#00d4ff", linewidth=2)
    ax.set_title("Temporal Importance", color="#e0e8f0", fontsize=12, fontweight=300)
    ax.set_xlabel("Time Step", color="#8ca5c8")
    ax.set_ylabel("Attention Weight", color="#8ca5c8")
    ax.tick_params(colors="#8ca5c8")
    ax.grid(True, alpha=0.1, color="#4da6ff")

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color("#1a3a5c")

    if title:
        fig.suptitle(title, color="#4da6ff", fontsize=16, fontweight=300, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Attention map saved to {output_path}")
