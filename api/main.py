"""
FastAPI application for MitraSETI.

Local-first API for radio-signal processing, ML classification, catalog
cross-referencing, and live WebSocket streaming during observation runs.
Adapted from the AstroLens API pattern.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from paths import (
    DB_PATH,
    FILTERBANK_DIR,
    MODELS_DIR,
    ensure_dirs,
)

from .database import SignalDB

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton instances
# ─────────────────────────────────────────────────────────────────────────────

_db: Optional[SignalDB] = None
_catalog_query = None
_pipeline = None
_live_connections: Set[WebSocket] = set()


def get_db() -> SignalDB:
    """Return the global SignalDB instance."""
    global _db
    if _db is None:
        _db = SignalDB()
    return _db


def get_catalog_query():
    """Lazy-load the RadioCatalogQuery."""
    global _catalog_query
    if _catalog_query is None:
        from catalog.radio_catalogs import RadioCatalogQuery

        _catalog_query = RadioCatalogQuery()
    return _catalog_query


def get_pipeline():
    """Lazy-load the MitraSETIPipeline."""
    global _pipeline
    if _pipeline is None:
        from pipeline import MitraSETIPipeline

        model_path = MODELS_DIR / "signal_classifier_v1.pt"
        ood_cal_path = MODELS_DIR / "ood_calibration.json"
        _pipeline = MitraSETIPipeline(
            model_path=str(model_path) if model_path.exists() else None,
            ood_calibration_path=str(ood_cal_path) if ood_cal_path.exists() else None,
        )
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """System health status."""

    status: str
    version: str
    gpu_available: bool
    models_loaded: bool
    disk_free_gb: float
    db_path: str


class SignalResponse(BaseModel):
    """A single detected signal."""

    id: int
    frequency_hz: float
    drift_rate: float
    snr: float
    rfi_score: float = 0.0
    classification: str = "unknown"
    confidence: float = 0.0
    is_candidate: bool = False
    is_verified: bool = False
    observation_id: Optional[int] = None
    detected_at: str = ""
    image_path: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    notes: Optional[str] = None


class SignalUpdate(BaseModel):
    """Allowed fields for PATCH /signals/{id}."""

    classification: Optional[str] = None
    confidence: Optional[float] = None
    rfi_score: Optional[float] = None
    is_candidate: Optional[bool] = None
    is_verified: Optional[bool] = None
    notes: Optional[str] = None


class ProcessRequest(BaseModel):
    """Metadata to accompany a filterbank upload."""

    source_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None


class ProcessResponse(BaseModel):
    """Result after processing a filterbank file."""

    observation_id: int
    file_path: str
    total_signals: int
    candidates_found: int
    status: str


class StatsResponse(BaseModel):
    """Aggregate processing statistics."""

    total_signals: int
    total_candidates: int
    verified_signals: int
    total_observations: int
    average_snr: float


class CatalogCrossRefResponse(BaseModel):
    """Result of catalog cross-reference lookup."""

    ra: float
    dec: float
    is_known_source: bool
    source_description: str
    results: List[Dict[str, Any]] = Field(default_factory=list)


class AstroLensCrossRefResponse(BaseModel):
    """Result of AstroLens optical cross-reference."""

    ra: float
    dec: float
    matches_found: int
    results: List[Dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    ensure_dirs()
    db = get_db()
    await db.init_db()
    logger.info("MitraSETI API started – database at %s", DB_PATH)
    yield
    logger.info("MitraSETI API shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MitraSETI API",
    description=(
        "Radio-signal processing pipeline for SETI. "
        "Detect, classify, and cross-reference narrowband signals."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Root & Health
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health: GPU, models, disk space."""
    gpu_available = False
    try:
        from inference.gpu_utils import DeviceInfo

        info = DeviceInfo.detect()
        gpu_available = info.device_type != "cpu"
    except Exception:
        pass

    models_loaded = any(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else False

    disk = shutil.disk_usage(str(DB_PATH.parent))
    disk_free_gb = round(disk.free / (1024**3), 2)

    return HealthResponse(
        status="ok",
        version="0.1.0",
        gpu_available=gpu_available,
        models_loaded=models_loaded,
        disk_free_gb=disk_free_gb,
        db_path=str(DB_PATH),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/process", response_model=ProcessResponse)
async def process_filterbank(
    file: UploadFile = File(...),
    source_name: Optional[str] = Query(None),
    ra: Optional[float] = Query(None),
    dec: Optional[float] = Query(None),
):
    """
    Upload and process a filterbank (.fil / .h5) file.

    Runs the full pipeline: hit detection → feature extraction →
    classification → RFI scoring → candidate promotion.
    """
    # Validate file type
    allowed_exts = {".fil", ".h5", ".hdf5", ".filterbank"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {allowed_exts}",
        )

    # Save uploaded file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    file_path = FILTERBANK_DIR / safe_name

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Create observation record
    db = get_db()
    obs_id = await db.add_observation(
        {
            "file_path": str(file_path),
            "source_name": source_name or Path(file.filename or "").stem,
            "ra": ra,
            "dec": dec,
            "status": "processing",
        }
    )

    # ── Run pipeline ────────────────────────────────────────────────
    total_signals = 0
    candidates_found = 0

    try:
        pipe = get_pipeline()
        result = pipe.process_file(str(file_path))

        summary = result.get("summary", {})
        total_signals = summary.get("total_hits_filtered", 0)
        candidates_found = summary.get("candidate_count", 0)

        for cand in result.get("candidates", []):
            signal_data = {
                "frequency_hz": cand.get("frequency_hz", 0),
                "drift_rate": cand.get("drift_rate", 0),
                "snr": cand.get("snr", 0),
                "rfi_score": cand.get("rfi_probability", 0),
                "classification": cand.get("classification", "unknown"),
                "confidence": cand.get("confidence", 0),
                "is_candidate": cand.get("is_candidate", False),
                "observation_id": obs_id,
                "ra": ra,
                "dec": dec,
            }
            await db.add_signal(signal_data)

        await db.update_observation(
            obs_id,
            {
                "total_signals": total_signals,
                "candidates_found": candidates_found,
                "status": "complete",
            },
        )
    except Exception as exc:
        await db.update_observation(
            obs_id,
            {
                "status": "error",
                "error_message": str(exc),
            },
        )
        raise HTTPException(500, f"Processing failed: {exc}")

    # Broadcast to live WebSocket clients
    await _broadcast(
        {
            "type": "observation_complete",
            "observation_id": obs_id,
            "total_signals": total_signals,
            "candidates_found": candidates_found,
        }
    )

    return ProcessResponse(
        observation_id=obs_id,
        file_path=str(file_path),
        total_signals=total_signals,
        candidates_found=candidates_found,
        status="complete",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signals CRUD
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/signals", response_model=List[SignalResponse])
async def list_signals(
    min_snr: Optional[float] = Query(None, description="Minimum SNR"),
    max_rfi_score: Optional[float] = Query(None, description="Maximum RFI score"),
    min_drift_rate: Optional[float] = Query(None, description="Min drift rate (Hz/s)"),
    max_drift_rate: Optional[float] = Query(None, description="Max drift rate (Hz/s)"),
    classification: Optional[str] = Query(None, description="Signal classification label"),
    is_candidate: Optional[bool] = Query(None, description="ET candidate flag"),
    observation_id: Optional[int] = Query(None, description="Filter by observation"),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """List detected signals with optional filters."""
    db = get_db()
    rows = await db.get_signals(
        min_snr=min_snr,
        max_rfi_score=max_rfi_score,
        min_drift_rate=min_drift_rate,
        max_drift_rate=max_drift_rate,
        classification=classification,
        is_candidate=is_candidate,
        observation_id=observation_id,
        limit=limit,
        offset=offset,
    )
    return [SignalResponse(**_bool_fields(r)) for r in rows]


@app.get("/signals/{signal_id}", response_model=SignalResponse)
async def get_signal(signal_id: int):
    """Get a single signal by ID."""
    db = get_db()
    row = await db.get_signal(signal_id)
    if not row:
        raise HTTPException(404, "Signal not found")
    return SignalResponse(**_bool_fields(row))


@app.patch("/signals/{signal_id}", response_model=SignalResponse)
async def update_signal(signal_id: int, update: SignalUpdate):
    """Update (verify / reject) a signal."""
    db = get_db()
    existing = await db.get_signal(signal_id)
    if not existing:
        raise HTTPException(404, "Signal not found")

    changes: Dict[str, Any] = {}
    for field_name, value in update.model_dump(exclude_unset=True).items():
        if value is not None:
            changes[field_name] = value

    if not changes:
        return SignalResponse(**_bool_fields(existing))

    updated = await db.update_signal(signal_id, changes)
    return SignalResponse(**_bool_fields(updated))


# ─────────────────────────────────────────────────────────────────────────────
# Candidates
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/candidates", response_model=List[SignalResponse])
async def list_candidates(
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """List signals promoted to ET candidate status, ordered by SNR."""
    db = get_db()
    rows = await db.get_candidates(limit=limit, offset=offset)
    return [SignalResponse(**_bool_fields(r)) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Processing statistics across all observations."""
    db = get_db()
    stats = await db.get_stats()
    return StatsResponse(**stats)


# ─────────────────────────────────────────────────────────────────────────────
# Catalog Cross-Reference
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/catalog/crossref", response_model=CatalogCrossRefResponse)
async def catalog_crossref(
    ra: float = Query(..., description="Right ascension (degrees, J2000)"),
    dec: float = Query(..., description="Declination (degrees, J2000)"),
    radius_arcmin: float = Query(5.0, description="Search radius in arcminutes"),
):
    """
    Cross-reference a sky position against radio astronomy catalogs
    (SIMBAD, NVSS, FIRST, ATNF Pulsar Catalogue).
    """
    from dataclasses import asdict

    from catalog.radio_catalogs import CatalogResult

    query = get_catalog_query()
    is_known, description = query.is_known_source(ra, dec, freq_mhz=0, radius_arcmin=radius_arcmin)

    # Collect all results for the detailed list
    all_results: List[CatalogResult] = []
    all_results.extend(query.query_simbad(ra, dec, radius_arcmin))
    all_results.extend(query.query_nvss(ra, dec, radius_arcmin))
    all_results.extend(query.query_first(ra, dec, radius_arcmin))
    all_results.extend(query.query_pulsar_catalog(ra, dec, radius_arcmin))

    return CatalogCrossRefResponse(
        ra=ra,
        dec=dec,
        is_known_source=is_known,
        source_description=description,
        results=[asdict(r) for r in all_results],
    )


# ─────────────────────────────────────────────────────────────────────────────
# AstroLens Optical Cross-Reference
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/astrolens/crossref", response_model=AstroLensCrossRefResponse)
async def astrolens_optical_crossref(
    ra: float = Query(..., description="Right ascension (degrees, J2000)"),
    dec: float = Query(..., description="Declination (degrees, J2000)"),
    radius_arcmin: float = Query(2.0, description="Search radius in arcminutes"),
):
    """
    Check if AstroLens has detected optical anomalies near this radio position.
    """
    from dataclasses import asdict

    from catalog.sky_position import astrolens_crossref as _xref

    matches = _xref(ra, dec, radius_arcmin=radius_arcmin)
    return AstroLensCrossRefResponse(
        ra=ra,
        dec=dec,
        matches_found=len(matches),
        results=[asdict(m) for m in matches],
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket – Live Signal Stream
# ─────────────────────────────────────────────────────────────────────────────


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Live signal stream during processing.

    Clients connect and receive JSON messages whenever new signals are
    detected or observations complete.
    """
    await ws.accept()
    _live_connections.add(ws)
    logger.info("WebSocket client connected (%d total)", len(_live_connections))
    try:
        while True:
            # Keep connection alive; clients send pings or text
            data = await ws.receive_text()
            # Echo heartbeat
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        _live_connections.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(_live_connections))


async def _broadcast(message: Dict[str, Any]) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    if not _live_connections:
        return
    payload = json.dumps(message)
    stale: List[WebSocket] = []
    for ws in _live_connections:
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _live_connections.discard(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bool_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """SQLite stores booleans as 0/1 – convert for Pydantic."""
    out = dict(row)
    for key in ("is_candidate", "is_verified"):
        if key in out:
            out[key] = bool(out[key])
    return out
