#!/usr/bin/env python3
"""
Synthetic Training Data Generator for MitraSETI

Generates ~5000 synthetic spectrograms (256 freq x 64 time) across 9 signal
classes, plus 10 Sigproc-format .fil files for testing the Rust core pipeline.

Signal classes:
    0: narrowband_drifting   – narrowband signal with non-zero drift rate
    1: narrowband_stationary – narrowband signal with zero drift
    2: broadband             – wide-band emission
    3: pulsed                – periodic on/off signal
    4: chirp                 – frequency-sweeping signal
    5: rfi_terrestrial       – strong, zero-drift, broadband (WiFi/GPS-like)
    6: rfi_satellite         – periodic, narrowband, multi-frequency
    7: noise                 – pure Gaussian/chi-squared noise
    8: candidate_et          – weak narrowband drifting with specific drift characteristics

Usage:
    python scripts/generate_training_data.py --output-dir data/training --count 600 --seed 42
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import chirp as scipy_chirp
from tqdm import tqdm

# Spectrogram dimensions
N_FREQ = 256
N_TIME = 64

# Class definitions (must match inference.signal_classifier.SignalType)
CLASS_NAMES = [
    "narrowband_drifting",
    "narrowband_stationary",
    "broadband",
    "pulsed",
    "chirp",
    "rfi_terrestrial",
    "rfi_satellite",
    "noise",
    "candidate_et",
]

DEFAULT_COUNTS = {
    "narrowband_drifting": 600,
    "narrowband_stationary": 600,
    "broadband": 600,
    "pulsed": 600,
    "chirp": 500,
    "rfi_terrestrial": 500,
    "rfi_satellite": 500,
    "noise": 600,
    "candidate_et": 500,
}


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _base_noise(rng: np.random.Generator, n_freq: int = N_FREQ, n_time: int = N_TIME) -> np.ndarray:
    """Generate base Gaussian noise spectrogram."""
    return rng.normal(0.0, 1.0, (n_freq, n_time))


def _inject_narrowband_line(
    spec: np.ndarray,
    rng: np.random.Generator,
    freq_start: float,
    drift_pixels: float,
    snr: float,
    width: float = 1.5,
) -> np.ndarray:
    """Draw a drifting narrowband line across the spectrogram."""
    n_freq, n_time = spec.shape
    for t in range(n_time):
        freq_center = freq_start + drift_pixels * t / n_time
        freq_idx = np.arange(n_freq)
        profile = snr * np.exp(-0.5 * ((freq_idx - freq_center) / width) ** 2)
        spec[:, t] += profile
    return spec


def generate_narrowband_drifting(rng: np.random.Generator) -> np.ndarray:
    freq_start = rng.uniform(30, N_FREQ - 30)
    drift = rng.choice([-1, 1]) * rng.uniform(5, 40)
    snr = rng.uniform(3.0, 15.0)
    width = rng.uniform(0.8, 2.5)
    spec = _base_noise(rng)
    return _inject_narrowband_line(spec, rng, freq_start, drift, snr, width)


def generate_narrowband_stationary(rng: np.random.Generator) -> np.ndarray:
    freq_center = rng.uniform(20, N_FREQ - 20)
    snr = rng.uniform(4.0, 20.0)
    width = rng.uniform(0.5, 2.0)
    spec = _base_noise(rng)
    return _inject_narrowband_line(spec, rng, freq_center, 0.0, snr, width)


def generate_broadband(rng: np.random.Generator) -> np.ndarray:
    spec = _base_noise(rng)
    center = rng.uniform(40, N_FREQ - 40)
    bw = rng.uniform(20, 80)
    snr = rng.uniform(2.0, 8.0)
    freq_idx = np.arange(N_FREQ)
    envelope = snr * np.exp(-0.5 * ((freq_idx - center) / (bw / 2.355)) ** 2)
    time_var = 1.0 + 0.3 * rng.normal(0, 1, N_TIME)
    spec += envelope[:, None] * time_var[None, :]
    return spec


def generate_pulsed(rng: np.random.Generator) -> np.ndarray:
    spec = _base_noise(rng)
    freq_center = rng.uniform(30, N_FREQ - 30)
    width = rng.uniform(0.8, 2.5)
    snr = rng.uniform(4.0, 15.0)
    period = rng.integers(4, 16)
    duty_cycle = rng.uniform(0.2, 0.6)
    on_duration = max(1, int(period * duty_cycle))
    freq_idx = np.arange(N_FREQ)
    profile = snr * np.exp(-0.5 * ((freq_idx - freq_center) / width) ** 2)
    for t in range(N_TIME):
        if (t % period) < on_duration:
            spec[:, t] += profile
    return spec


def generate_chirp(rng: np.random.Generator) -> np.ndarray:
    spec = _base_noise(rng)
    f0 = rng.uniform(30, 100)
    f1 = rng.uniform(N_FREQ - 100, N_FREQ - 30)
    if rng.random() < 0.5:
        f0, f1 = f1, f0
    snr = rng.uniform(3.0, 10.0)
    width = rng.uniform(1.0, 3.0)
    for t in range(N_TIME):
        frac = t / (N_TIME - 1)
        freq_center = f0 + (f1 - f0) * frac
        freq_idx = np.arange(N_FREQ)
        profile = snr * np.exp(-0.5 * ((freq_idx - freq_center) / width) ** 2)
        spec[:, t] += profile
    return spec


def generate_rfi_terrestrial(rng: np.random.Generator) -> np.ndarray:
    spec = _base_noise(rng)
    center = rng.uniform(40, N_FREQ - 40)
    bw = rng.uniform(15, 60)
    snr = rng.uniform(10.0, 50.0)
    freq_idx = np.arange(N_FREQ)
    envelope = snr * np.exp(-0.5 * ((freq_idx - center) / (bw / 2.355)) ** 2)
    spec += envelope[:, None] * np.ones((1, N_TIME))
    n_harmonics = rng.integers(0, 4)
    for _ in range(n_harmonics):
        h_center = rng.uniform(10, N_FREQ - 10)
        h_snr = snr * rng.uniform(0.1, 0.4)
        h_bw = rng.uniform(5, 20)
        h_env = h_snr * np.exp(-0.5 * ((freq_idx - h_center) / (h_bw / 2.355)) ** 2)
        spec += h_env[:, None]
    return spec


def generate_rfi_satellite(rng: np.random.Generator) -> np.ndarray:
    spec = _base_noise(rng)
    n_freqs = rng.integers(2, 6)
    period = rng.integers(4, 12)
    duty = rng.uniform(0.3, 0.7)
    on_dur = max(1, int(period * duty))
    for _ in range(n_freqs):
        fc = rng.uniform(15, N_FREQ - 15)
        snr = rng.uniform(5.0, 25.0)
        width = rng.uniform(0.5, 2.0)
        freq_idx = np.arange(N_FREQ)
        profile = snr * np.exp(-0.5 * ((freq_idx - fc) / width) ** 2)
        for t in range(N_TIME):
            if (t % period) < on_dur:
                spec[:, t] += profile
    return spec


def generate_noise(rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        return rng.normal(0.0, 1.0, (N_FREQ, N_TIME))
    df = rng.integers(2, 8)
    return rng.chisquare(df, (N_FREQ, N_TIME)) - df


def generate_candidate_et(rng: np.random.Generator) -> np.ndarray:
    """Weak narrowband drifting with specific drift rate characteristics.

    ET candidates are characterized by:
    - Low SNR (harder to detect)
    - Drift rate consistent with exoplanetary rotation (~0.1-4 Hz/s equivalent)
    - Very narrow bandwidth
    """
    spec = _base_noise(rng)
    freq_start = rng.uniform(40, N_FREQ - 40)
    drift = rng.choice([-1, 1]) * rng.uniform(2, 15)
    snr = rng.uniform(1.5, 5.0)
    width = rng.uniform(0.3, 1.2)
    return _inject_narrowband_line(spec, rng, freq_start, drift, snr, width)


GENERATORS = {
    "narrowband_drifting": generate_narrowband_drifting,
    "narrowband_stationary": generate_narrowband_stationary,
    "broadband": generate_broadband,
    "pulsed": generate_pulsed,
    "chirp": generate_chirp,
    "rfi_terrestrial": generate_rfi_terrestrial,
    "rfi_satellite": generate_rfi_satellite,
    "noise": generate_noise,
    "candidate_et": generate_candidate_et,
}


# ---------------------------------------------------------------------------
# Sigproc filterbank writer
# ---------------------------------------------------------------------------

def _write_sigproc_string(f, keyword: str) -> None:
    """Write a Sigproc header keyword string (length-prefixed)."""
    encoded = keyword.encode("ascii")
    f.write(struct.pack("I", len(encoded)))
    f.write(encoded)


def _write_sigproc_header_value(f, keyword: str, value, fmt: str) -> None:
    """Write a keyword-value pair in Sigproc header format."""
    _write_sigproc_string(f, keyword)
    f.write(struct.pack(fmt, value))


def write_filterbank_file(
    filepath: Path,
    data: np.ndarray,
    fch1: float = 1420.0,
    foff: float = -0.00028,
    tsamp: float = 18.253611,
    src_name: str = "synthetic",
    nbits: int = 32,
) -> None:
    """Write a Sigproc-format filterbank (.fil) file.

    Args:
        filepath: Output path.
        data: 2D array of shape (n_time, n_channels), float32.
        fch1: Frequency of first channel (MHz).
        foff: Channel bandwidth (MHz, negative = descending).
        tsamp: Sampling time (seconds).
        src_name: Source name string.
        nbits: Bits per sample (32 for float32).
    """
    n_time, n_chans = data.shape
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        _write_sigproc_string(f, "HEADER_START")

        _write_sigproc_string(f, "source_name")
        src_bytes = src_name.encode("ascii")
        f.write(struct.pack("I", len(src_bytes)))
        f.write(src_bytes)

        _write_sigproc_header_value(f, "fch1", fch1, "d")
        _write_sigproc_header_value(f, "foff", foff, "d")
        _write_sigproc_header_value(f, "nchans", n_chans, "i")
        _write_sigproc_header_value(f, "nbits", nbits, "i")
        _write_sigproc_header_value(f, "tsamp", tsamp, "d")
        _write_sigproc_header_value(f, "tstart", 59000.0, "d")
        _write_sigproc_header_value(f, "nifs", 1, "i")
        _write_sigproc_header_value(f, "src_raj", 0.0, "d")
        _write_sigproc_header_value(f, "src_dej", 0.0, "d")

        _write_sigproc_string(f, "HEADER_END")

        data.astype(np.float32).tofile(f)


def generate_filterbank_files(
    output_dir: Path,
    rng: np.random.Generator,
    n_files: int = 10,
    n_chans: int = 1024,
    n_time: int = 16,
) -> list[str]:
    """Generate synthetic .fil files for Rust core pipeline testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for i in tqdm(range(n_files), desc="Generating .fil files"):
        if i == 0:
            name = "synthetic_voyager.fil"
            n_signals = 1
            is_voyager = True
        else:
            name = f"synthetic_{i:03d}.fil"
            n_signals = rng.integers(1, 4)
            is_voyager = False

        noise = rng.normal(0, 1, (n_time, n_chans)).astype(np.float32)

        for s in range(n_signals):
            if is_voyager:
                chan_center = n_chans // 2
                drift_per_step = 2.5
                snr = 12.0
            else:
                chan_center = rng.integers(50, n_chans - 50)
                drift_per_step = rng.uniform(-5, 5)
                snr = rng.uniform(3.0, 20.0)

            width = rng.uniform(0.5, 2.0)
            chan_idx = np.arange(n_chans)
            for t in range(n_time):
                center = chan_center + drift_per_step * t / n_time
                profile = snr * np.exp(-0.5 * ((chan_idx - center) / width) ** 2)
                noise[t, :] += profile.astype(np.float32)

        fpath = output_dir / name
        write_filterbank_file(
            fpath,
            noise,
            fch1=1420.405751,
            foff=-0.00028,
            tsamp=18.253611,
            src_name="Voyager1" if is_voyager else f"Synthetic_{i:03d}",
        )
        generated.append(str(fpath))

    return generated


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_training_data(
    output_dir: Path,
    counts: dict[str, int],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate all training spectrograms and labels.

    Returns:
        spectrograms: array of shape (N, N_FREQ, N_TIME)
        labels: array of shape (N,) with integer class labels
        metadata: dict with generation parameters
    """
    rng = np.random.default_rng(seed)

    total = sum(counts.values())
    spectrograms = np.empty((total, N_FREQ, N_TIME), dtype=np.float32)
    labels = np.empty(total, dtype=np.int64)

    idx = 0
    for class_label, class_name in enumerate(CLASS_NAMES):
        n = counts.get(class_name, 0)
        if n == 0:
            continue
        gen_func = GENERATORS[class_name]
        for _ in tqdm(range(n), desc=f"  {class_name}", leave=False):
            spec = gen_func(rng).astype(np.float32)
            spectrograms[idx] = spec
            labels[idx] = class_label
            idx += 1

    spectrograms = spectrograms[:idx]
    labels = labels[:idx]

    shuffle = rng.permutation(idx)
    spectrograms = spectrograms[shuffle]
    labels = labels[shuffle]

    metadata = {
        "n_samples": int(idx),
        "n_freq": N_FREQ,
        "n_time": N_TIME,
        "class_names": CLASS_NAMES,
        "class_counts": {name: int(counts.get(name, 0)) for name in CLASS_NAMES},
        "seed": seed,
        "dtype": "float32",
    }

    return spectrograms, labels, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for MitraSETI signal classifier."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory for spectrograms/labels (default: data/training)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override samples per class (uses per-class defaults if not set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir

    if args.count is not None:
        counts = {name: args.count for name in CLASS_NAMES}
    else:
        counts = dict(DEFAULT_COUNTS)

    total = sum(counts.values())
    print(f"MitraSETI Training Data Generator")
    print(f"  Output:  {output_dir}")
    print(f"  Classes: {len(CLASS_NAMES)}")
    print(f"  Total:   {total} spectrograms ({N_FREQ}x{N_TIME})")
    print(f"  Seed:    {args.seed}")
    print()

    t0 = time.time()
    print("Generating spectrograms...")
    spectrograms, labels, metadata = generate_training_data(output_dir, counts, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "spectrograms.npy"
    labels_path = output_dir / "labels.npy"
    meta_path = output_dir / "metadata.json"

    print(f"\nSaving spectrograms → {spec_path}")
    np.save(spec_path, spectrograms)

    print(f"Saving labels       → {labels_path}")
    np.save(labels_path, labels)

    print(f"Saving metadata     → {meta_path}")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    fil_dir = project_root / "data" / "filterbank"
    print(f"\nGenerating filterbank files → {fil_dir}")
    rng = np.random.default_rng(args.seed + 1000)
    fil_paths = generate_filterbank_files(fil_dir, rng, n_files=10)

    metadata["filterbank_files"] = fil_paths
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Spectrograms: {spectrograms.shape} ({spec_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Labels:       {labels.shape}")
    print(f"  Filterbank:   {len(fil_paths)} files")
    for name in CLASS_NAMES:
        print(f"    {name:25s}: {counts.get(name, 0):5d} samples")


if __name__ == "__main__":
    main()
