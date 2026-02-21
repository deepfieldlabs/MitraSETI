"""
astroSETI Web Interface

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
API_BASE = os.environ.get("ASTROSETI_API", "http://localhost:9000")

# Setup app
app = FastAPI(title="astroSETI Web", version="1.0.0")

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
        logger.error(f"API error: {e}")

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
        logger.error(f"API error: {e}")

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
        logger.error(f"API error: {e}")

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
    parser = argparse.ArgumentParser(description="astroSETI Web Interface")
    parser.add_argument("--port", type=int, default=9090, help="Port (default: 9090)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"\n  astroSETI Web Interface")
    print(f"  http://localhost:{args.port}")
    print(f"  API Backend: {API_BASE}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
