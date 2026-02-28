"""
MitraSETI Web Interface

Browser-based UI for SETI signal analysis, waterfall viewing, RFI monitoring,
and sky mapping. Proxies requests to the main API backend.

Usage:
    python -m web.app          # Start web UI on port 9090
    python -m web.app --port 9090
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import httpx
import uvicorn

from paths import (
    DATA_DIR,
    CANDIDATES_DIR,
    STREAMING_STATE,
    DISCOVERY_STATE,
    CANDIDATES_FILE,
    ASTROLENS_CANDIDATES_FILE,
)

logger = logging.getLogger(__name__)

# API backend URL
API_BASE = os.environ.get("MITRASETI_API", "http://localhost:9000")

# Well-known target coordinates (RA, Dec in degrees) for sky map plotting
_TARGET_COORDS: dict[str, tuple[float, float]] = {
    "VOYAGER": (286.86, 12.17),
    "TRAPPIST": (346.62, -5.04),
    "TRAPPIST-1": (346.62, -5.04),
    "3C161": (93.0, -5.88),
    "3C286": (202.78, 30.51),
    "3C48": (24.42, 33.16),
    "HD-109376": (188.56, -26.89),
    "HD109376": (188.56, -26.89),
    "LHS292": (159.58, -44.32),
    "GJ699": (269.45, 4.69),
    "HIP107346": (326.20, 38.78),
    "KIC8462852": (301.56, 44.46),
    "KEPLER992B": (291.42, 42.33),
    "PROXCEN": (217.39, -62.68),
    "PROXIMA": (217.39, -62.68),
}

# Setup app
app = FastAPI(title="MitraSETI Web", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def api_client() -> httpx.Client:
    """Get API client."""
    return httpx.Client(base_url=API_BASE, timeout=30.0)


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

def _load_local_stats() -> dict:
    """Build stats dict from local state files."""
    stats = {}
    if STREAMING_STATE.exists():
        try:
            with open(STREAMING_STATE) as f:
                state = json.load(f)
            stats["total_signals"] = state.get("total_signals", 0)
            stats["total_candidates"] = state.get("total_candidates", 0)
            stats["total_rfi_rejected"] = state.get("total_rfi_rejected", 0)
            stats["uptime_seconds"] = state.get("uptime_seconds", 0)
            stats["cycles_completed"] = state.get("cycles_completed", 0)
            stats["avg_snr"] = state.get("avg_snr", 0)
            stats["max_snr"] = state.get("max_snr", 0)
            stats["classification_counts"] = state.get("classification_counts", {})
        except Exception:
            pass

    if DISCOVERY_STATE.exists():
        try:
            with open(DISCOVERY_STATE) as f:
                disc = json.load(f)
            stats["total_signals"] = disc.get("total_analyzed", stats.get("total_signals", 0))
            stats["total_candidates"] = disc.get("candidates_found", stats.get("total_candidates", 0))
            stats["total_rfi_rejected"] = disc.get("rfi_rejected", stats.get("total_rfi_rejected", 0))
            stats["cycles_completed"] = disc.get("cycles_completed", stats.get("cycles_completed", 0))
        except Exception:
            pass
    return stats


def _load_local_candidates() -> list:
    """Load candidates from verified_candidates.json."""
    if CANDIDATES_FILE.exists():
        try:
            with open(CANDIDATES_FILE) as f:
                cands = json.load(f)
            cands.sort(key=lambda c: c.get("snr", 0), reverse=True)
            return cands
        except Exception:
            pass
    return []


def _enrich_for_skymap(candidates: list) -> list:
    """Add ra_deg/dec_deg coordinates and normalize fields for the sky map."""
    enriched = []
    for i, c in enumerate(candidates):
        target = (c.get("target_name") or c.get("category") or "").upper()
        ra, dec = None, None
        for key, coords in _TARGET_COORDS.items():
            if key.upper() in target or target in key.upper():
                ra, dec = coords
                break
        if ra is None:
            continue

        enriched.append({
            "id": i + 1,
            "ra_deg": ra + (i * 0.05),
            "dec_deg": dec + (i * 0.03),
            "snr": c.get("snr", 0),
            "frequency_mhz": c.get("frequency_hz", 0) / 1e6,
            "drift_rate": c.get("drift_rate", 0),
            "rfi_score": c.get("rfi_probability", 0),
            "classification": "candidate" if c.get("classification", "") in (
                "narrowband_drifting", "candidate_et"
            ) else "rfi" if "rfi" in c.get("classification", "").lower() else "signal",
            "target_name": c.get("target_name", target),
            "file": c.get("file_name", ""),
            "ood_score": c.get("ood_score", 0),
            "confidence": c.get("confidence", 0),
        })
    return enriched


def _build_skymap_observations() -> list:
    """Build full observation list from streaming state and candidates."""
    observations = []
    seen_targets = set()

    # First add verified candidates with real data
    candidates = _load_local_candidates()
    observations.extend(_enrich_for_skymap(candidates))
    for o in observations:
        seen_targets.add(o.get("target_name", ""))

    # Then add category stats as aggregate observations
    if STREAMING_STATE.exists():
        try:
            with open(STREAMING_STATE) as f:
                state = json.load(f)
            cat_stats = state.get("category_stats", {})
            idx = len(observations) + 1
            for _cat, info in cat_stats.items():
                tgt = info.get("target_name", "")
                if tgt in seen_targets:
                    continue
                tgt_upper = tgt.upper()
                ra, dec = None, None
                for key, coords in _TARGET_COORDS.items():
                    if key.upper() in tgt_upper or tgt_upper in key.upper():
                        ra, dec = coords
                        break
                if ra is None:
                    continue

                n_signals = info.get("signals", 0)
                n_cands = info.get("candidates", 0)
                n_rfi = info.get("rfi", 0)

                cls = "candidate" if n_cands > 0 else "rfi" if n_rfi > n_signals * 0.5 else "signal"
                observations.append({
                    "id": idx,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "snr": max(n_cands * 5, 10) if n_cands > 0 else 5,
                    "frequency_mhz": 0,
                    "drift_rate": 0,
                    "rfi_score": n_rfi / max(n_signals, 1),
                    "classification": cls,
                    "target_name": tgt,
                    "file": f"{info.get('files', 0)} files",
                    "signals": n_signals,
                    "candidates": n_cands,
                    "description": info.get("description", ""),
                })
                idx += 1
        except Exception:
            pass
    return observations


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    stats = {}
    candidates = []
    health = {}

    try:
        with api_client() as client:
            r = client.get("/stats")
            if r.status_code == 200:
                stats = r.json()

            r = client.get("/signals", params={"limit": 20, "candidates_only": True})
            if r.status_code == 200:
                candidates = r.json()

            r = client.get("/health")
            if r.status_code == 200:
                health = r.json()
    except Exception as e:
        logger.warning(f"API unavailable, using local state files: {e}")

    if not stats:
        stats = _load_local_stats()

    if not candidates:
        candidates = _load_local_candidates()[:20]

    if not health:
        running = False
        if STREAMING_STATE.exists():
            age = time.time() - STREAMING_STATE.stat().st_mtime
            running = age < 120
        health = {"status": "ok" if running else "offline", "mode": "standalone"}

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "candidates": candidates,
        "health": health,
        "mode": "dashboard",
    })


@app.get("/waterfall", response_class=HTMLResponse)
async def waterfall_page(request: Request):
    """Waterfall spectrogram viewer page."""
    return templates.TemplateResponse("waterfall.html", {
        "request": request,
        "mode": "waterfall",
    })


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(
    request: Request,
    page: int = 0,
    limit: int = 24,
    classification: str = "",
    min_snr: float = 0,
    max_snr: float = 999,
    sort: str = "snr_desc",
):
    """Signal gallery page."""
    signals = []
    stats = {}

    try:
        with api_client() as client:
            params = {"skip": page * limit, "limit": limit}
            if classification:
                params["classification"] = classification
            if min_snr > 0:
                params["min_snr"] = min_snr
            if max_snr < 999:
                params["max_snr"] = max_snr
            params["sort"] = sort

            r = client.get("/signals", params=params)
            if r.status_code == 200:
                signals = r.json()

            r = client.get("/stats")
            if r.status_code == 200:
                stats = r.json()
    except Exception as e:
        logger.warning(f"API unavailable, using local state files: {e}")

    if not signals:
        all_signals = _load_local_candidates()

        if classification:
            all_signals = [
                s for s in all_signals
                if s.get("classification", "").lower() == classification.lower()
            ]
        if min_snr > 0:
            all_signals = [s for s in all_signals if s.get("snr", 0) >= min_snr]
        if max_snr < 999:
            all_signals = [s for s in all_signals if s.get("snr", 0) <= max_snr]

        if sort == "snr_desc":
            all_signals.sort(key=lambda s: s.get("snr", 0), reverse=True)
        elif sort == "snr_asc":
            all_signals.sort(key=lambda s: s.get("snr", 0))
        elif sort == "freq_desc":
            all_signals.sort(key=lambda s: s.get("frequency", 0), reverse=True)
        elif sort == "freq_asc":
            all_signals.sort(key=lambda s: s.get("frequency", 0))
        elif sort == "time_desc":
            all_signals.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
        elif sort == "time_asc":
            all_signals.sort(key=lambda s: s.get("timestamp", ""))

        start = page * limit
        signals = all_signals[start : start + limit]

    if not stats:
        stats = _load_local_stats()

    return templates.TemplateResponse("signals.html", {
        "request": request,
        "signals": signals,
        "stats": stats,
        "page": page,
        "limit": limit,
        "classification": classification,
        "min_snr": min_snr,
        "max_snr": max_snr,
        "sort": sort,
        "mode": "signals",
    })


@app.get("/rfi", response_class=HTMLResponse)
async def rfi_page(request: Request):
    """RFI dashboard page."""
    rfi_stats = {}

    try:
        with api_client() as client:
            r = client.get("/rfi/stats")
            if r.status_code == 200:
                rfi_stats = r.json()
    except Exception as e:
        logger.error(f"API error: {e}")

    return templates.TemplateResponse("rfi.html", {
        "request": request,
        "rfi_stats": rfi_stats,
        "rfi_stats_json": json.dumps(rfi_stats),
        "mode": "rfi",
    })


@app.get("/streaming", response_class=HTMLResponse)
async def streaming_page(request: Request):
    """Streaming monitor dashboard with live charts."""
    streaming = {}
    snapshots = []

    # Load streaming state
    if STREAMING_STATE.exists():
        try:
            with open(STREAMING_STATE) as f:
                streaming = json.load(f)
                snapshots = streaming.get("daily_snapshots", [])
        except Exception:
            pass

    # Check if still running
    if STREAMING_STATE.exists():
        age = time.time() - STREAMING_STATE.stat().st_mtime
        streaming["running"] = age < 120 and not streaming.get("completed", False)
    else:
        streaming["running"] = False

    # Merge live discovery state
    if DISCOVERY_STATE.exists():
        try:
            with open(DISCOVERY_STATE) as f:
                disc = json.load(f)
            streaming["total_signals"] = disc.get("total_analyzed", streaming.get("total_signals", 0))
            streaming["total_candidates"] = disc.get("candidates_found", streaming.get("total_candidates", 0))
            streaming["live_cycles"] = disc.get("cycles_completed", 0)
        except Exception:
            pass

    # Load top candidates
    if CANDIDATES_FILE.exists():
        try:
            with open(CANDIDATES_FILE) as f:
                cands = json.load(f)
            cands.sort(key=lambda c: c.get("snr", 0), reverse=True)
            streaming["top_candidates"] = cands[:20]
        except Exception:
            pass

    return templates.TemplateResponse("streaming.html", {
        "request": request,
        "streaming": streaming,
        "snapshots_json": json.dumps(snapshots),
        "now": datetime.now().strftime("%H:%M:%S"),
        "mode": "streaming",
    })


@app.get("/skymap", response_class=HTMLResponse)
async def skymap_page(request: Request):
    """Interactive sky map page."""
    observations = []
    astrolens_matches = []

    try:
        with api_client() as client:
            r = client.get("/signals", params={"limit": 500})
            if r.status_code == 200:
                observations = r.json()
    except Exception as e:
        logger.warning(f"API unavailable, using local state files: {e}")

    if not observations:
        observations = _build_skymap_observations()

    # Load AstroLens cross-matches
    if ASTROLENS_CANDIDATES_FILE.exists():
        try:
            with open(ASTROLENS_CANDIDATES_FILE) as f:
                astrolens_matches = json.load(f)
        except Exception:
            pass

    return templates.TemplateResponse("skymap.html", {
        "request": request,
        "observations_json": json.dumps(observations),
        "astrolens_json": json.dumps(astrolens_matches),
        "mode": "skymap",
    })


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Reports summary page."""
    stats = _load_local_stats()
    candidates = _load_local_candidates()
    category_stats = {}
    pipeline_metrics = {}

    if STREAMING_STATE.exists():
        try:
            with open(STREAMING_STATE) as f:
                state = json.load(f)
            category_stats = state.get("category_stats", {})
            pipeline_metrics = state.get("pipeline_metrics", {})
            stats["total_files_processed"] = state.get("total_files_processed", 0)
            stats["total_runtime_hours"] = state.get("total_runtime_hours", 0)
            stats["total_ood_anomalies"] = state.get("total_ood_anomalies", 0)
            stats["current_mode"] = state.get("current_mode", "unknown")
            stats["completed"] = state.get("completed", False)
        except Exception:
            pass

    return templates.TemplateResponse("reports.html", {
        "request": request,
        "stats": stats,
        "candidates": candidates[:20],
        "category_stats": category_stats,
        "category_stats_json": json.dumps(category_stats),
        "pipeline_metrics": pipeline_metrics,
        "mode": "reports",
    })


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About MitraSETI page."""
    return templates.TemplateResponse("about.html", {
        "request": request,
        "mode": "about",
    })


# ─────────────────────────────────────────────────────────────────────────────
# API Proxy Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/signals")
async def proxy_signals(limit: int = 100, skip: int = 0):
    """Proxy signals from backend."""
    try:
        with api_client() as client:
            r = client.get("/signals", params={"limit": limit, "skip": skip})
            return r.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stats")
async def proxy_stats():
    """Proxy stats from backend."""
    try:
        with api_client() as client:
            r = client.get("/stats")
            return r.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/streaming-state")
async def streaming_state():
    """Get live streaming state for dashboard auto-refresh."""
    data = {}

    if STREAMING_STATE.exists():
        try:
            with open(STREAMING_STATE) as f:
                data = json.load(f)
            age = time.time() - STREAMING_STATE.stat().st_mtime
            data["running"] = age < 120 and not data.get("completed", False)
        except Exception as e:
            return {"error": str(e), "running": False}

    if DISCOVERY_STATE.exists():
        try:
            with open(DISCOVERY_STATE) as f:
                disc = json.load(f)
            data["live_signals"] = disc.get("total_analyzed", 0)
            data["live_candidates"] = disc.get("candidates_found", 0)
            data["live_cycles"] = disc.get("cycles_completed", 0)
            data["live_rfi_rejected"] = disc.get("rfi_rejected", 0)
            disc_age = time.time() - DISCOVERY_STATE.stat().st_mtime
            if disc_age < 120:
                data["running"] = True
        except Exception:
            pass

    if CANDIDATES_FILE.exists():
        try:
            with open(CANDIDATES_FILE) as f:
                cands = json.load(f)
            cands.sort(key=lambda c: c.get("snr", 0), reverse=True)
            data["top_candidates"] = cands[:20]
            data["total_verified"] = len(cands)
        except Exception:
            pass

    if not data:
        return {"running": False, "started_at": ""}
    return data


@app.get("/api/rfi-stats")
async def rfi_stats():
    """Get RFI statistics."""
    try:
        with api_client() as client:
            r = client.get("/rfi/stats")
            return r.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/device")
async def device_info():
    """Get GPU/device information."""
    try:
        from inference.gpu_utils import DeviceInfo
        info = DeviceInfo.detect()
        return info.to_dict()
    except Exception as e:
        return {"device_type": "unknown", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="MitraSETI Web Interface")
    parser.add_argument("--port", type=int, default=9090, help="Port (default: 9090)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"\n  MitraSETI Web Interface")
    print(f"  http://localhost:{args.port}")
    print(f"  API Backend: {API_BASE}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
