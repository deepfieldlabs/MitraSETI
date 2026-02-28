"""
Centralized path configuration for MitraSETI.

All large files (data, models, database) are stored OUTSIDE the git repo
in a sibling folder called 'mitraseti_artifacts' to avoid bloating the repo.
(The directory on disk is still called 'mitraseti_artifacts' — unchanged.)

Directory Structure:
    projects/
    ├── MitraSETI/                 <- This git repo (code only)
    │   ├── api/
    │   ├── catalog/
    │   ├── inference/
    │   └── ...
    └── mitraseti_artifacts/       <- Large files (gitignored)
        ├── data/                  <- SQLite DB, filterbank files, plots
        ├── models/                <- Trained model weights
        └── candidates/            <- Verified ET candidate exports

Environment Variables (override defaults):
    MITRASETI_ARTIFACTS_DIR  - Base path for all artifacts
    DATABASE_URL             - SQLite database URL
    MODELS_DIR               - Trained model weights
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Base Paths
# ─────────────────────────────────────────────────────────────────────────────

# Project root (where this file lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Artifacts directory (sibling to project, contains all large files)
# Default: ../mitraseti_artifacts relative to project root (dir name on disk unchanged)
ARTIFACTS_DIR = Path(
    os.environ.get(
        "MITRASETI_ARTIFACTS_DIR",
        PROJECT_ROOT.parent / "mitraseti_artifacts",
    )
).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Data Paths (runtime data – database, filterbank files, plots)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("DATA_DIR", ARTIFACTS_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database
DB_PATH = DATA_DIR / "mitraseti.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite+aiosqlite:///{DB_PATH}")

# Filterbank / HDF5 input files
# Primary: real Breakthrough Listen data (manually downloaded)
BL_DATA_DIR = DATA_DIR / "breakthrough_listen_data_files"
# Fallback: synthetic filterbank directory
_SYNTHETIC_DIR = DATA_DIR / "filterbank"

FILTERBANK_DIR = Path(
    os.environ.get(
        "FILTERBANK_DIR",
        str(BL_DATA_DIR) if BL_DATA_DIR.exists() else str(_SYNTHETIC_DIR),
    )
)
FILTERBANK_DIR.mkdir(parents=True, exist_ok=True)

# Waterfall plot images
PLOTS_DIR = Path(os.environ.get("PLOTS_DIR", DATA_DIR / "plots"))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Weights
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ARTIFACTS_DIR / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Candidates & State Files
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES_DIR = Path(os.environ.get("CANDIDATES_DIR", ARTIFACTS_DIR / "candidates"))
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

# Persistent state files
STREAMING_STATE = DATA_DIR / "streaming_state.json"
DISCOVERY_STATE = DATA_DIR / "discovery_state.json"
CANDIDATES_FILE = CANDIDATES_DIR / "verified_candidates.json"

# AstroLens cross-reference (sibling project)
ASTROLENS_ARTIFACTS_DIR = Path(
    os.environ.get(
        "ASTROLENS_ARTIFACTS_DIR",
        PROJECT_ROOT.parent / "astrolens_artifacts",
    )
)
ASTROLENS_CANDIDATES_FILE = ASTROLENS_ARTIFACTS_DIR / "data" / "anomaly_candidates.json"


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dirs() -> None:
    """Ensure all required directories exist."""
    for d in [DATA_DIR, FILTERBANK_DIR, PLOTS_DIR, MODELS_DIR, CANDIDATES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_model_path(model_name: str = "signal_classifier") -> Path:
    """Get path to a specific model checkpoint."""
    return MODELS_DIR / model_name


def get_plot_path(signal_id: int, suffix: str = "waterfall") -> Path:
    """Get path for a signal's waterfall plot."""
    return PLOTS_DIR / f"signal_{signal_id}_{suffix}.png"


# Print paths on import (for debugging)
if __name__ == "__main__":
    print("MitraSETI Paths Configuration")
    print("=" * 50)
    print(f"PROJECT_ROOT:        {PROJECT_ROOT}")
    print(f"ARTIFACTS_DIR:       {ARTIFACTS_DIR}")
    print(f"DATA_DIR:            {DATA_DIR}")
    print(f"DB_PATH:             {DB_PATH}")
    print(f"DATABASE_URL:        {DATABASE_URL}")
    print(f"FILTERBANK_DIR:      {FILTERBANK_DIR}")
    print(f"PLOTS_DIR:           {PLOTS_DIR}")
    print(f"MODELS_DIR:          {MODELS_DIR}")
    print(f"CANDIDATES_DIR:      {CANDIDATES_DIR}")
    print(f"STREAMING_STATE:     {STREAMING_STATE}")
    print(f"DISCOVERY_STATE:     {DISCOVERY_STATE}")
    print(f"CANDIDATES_FILE:     {CANDIDATES_FILE}")
    print(f"ASTROLENS_CANDIDATES: {ASTROLENS_CANDIDATES_FILE}")
