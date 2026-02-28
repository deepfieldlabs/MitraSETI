# API Reference

MitraSETI exposes a REST API via FastAPI. The interactive Swagger UI documentation is available at `http://localhost:8000/docs` when the server is running.

## Starting the Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Endpoints

### `GET /health`

System health check. Returns GPU status, loaded models, disk space, and overall readiness.

**Response:**

```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_device": "mps",
  "model_loaded": true,
  "ood_calibrated": true,
  "disk_free_gb": 142.3,
  "rust_core_available": true
}
```

**Example:**

```bash
curl http://localhost:8000/health
```

---

### `POST /process`

Upload and process a `.fil` or `.h5` file through the full pipeline (de-Doppler search, RFI filtering, ML classification, OOD detection, catalog cross-reference).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | Filterbank file (.fil or .h5) |
| `source_name` | string | No | Source name (e.g., "Voyager-1") |
| `ra` | float | No | Right ascension in degrees |
| `dec` | float | No | Declination in degrees |

**Response:**

```json
{
  "observation_id": 1,
  "file": "voyager1.h5",
  "source_name": "Voyager-1",
  "total_signals": 847,
  "candidates_found": 1,
  "rfi_rejected": 824,
  "processing_time_ms": 60,
  "candidates": [
    {
      "id": 1,
      "frequency_hz": 8419921066.0,
      "drift_rate": 0.3928,
      "snr": 245.7,
      "classification": "narrowband_drifting",
      "confidence": 0.982,
      "rfi_score": 0.05,
      "ood_score": 0.12,
      "is_anomaly": false
    }
  ]
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/process \
    -F "file=@data/voyager1.h5" \
    -F "source_name=Voyager-1" \
    -F "ra=286.86" \
    -F "dec=12.17"
```

---

### `GET /signals`

List detected signals with optional filters. Returns paginated results ordered by SNR (descending).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_snr` | float | — | Minimum SNR filter |
| `max_snr` | float | — | Maximum SNR filter |
| `min_drift` | float | — | Minimum drift rate (Hz/s) |
| `max_drift` | float | — | Maximum drift rate (Hz/s) |
| `classification` | string | — | Filter by signal class (e.g., "narrowband_drifting") |
| `is_candidate` | bool | — | Filter to candidates only |
| `observation_id` | int | — | Filter by observation |
| `limit` | int | 100 | Maximum results to return |
| `offset` | int | 0 | Pagination offset |

**Response:**

```json
{
  "signals": [
    {
      "id": 1,
      "frequency_hz": 8419921066.0,
      "drift_rate": 0.3928,
      "snr": 245.7,
      "rfi_score": 0.05,
      "classification": "narrowband_drifting",
      "confidence": 0.982,
      "is_candidate": true,
      "is_verified": false,
      "observation_id": 1,
      "detected_at": "2025-01-15T08:00:32",
      "ra": 286.86,
      "dec": 12.17,
      "bandwidth_hz": 2.8,
      "notes": null
    }
  ],
  "total": 847,
  "limit": 100,
  "offset": 0
}
```

**Examples:**

```bash
# All signals with SNR > 20
curl "http://localhost:8000/signals?min_snr=20"

# Only candidates
curl "http://localhost:8000/signals?is_candidate=true"

# Narrowband drifting signals from a specific observation
curl "http://localhost:8000/signals?classification=narrowband_drifting&observation_id=1"

# Paginate through results
curl "http://localhost:8000/signals?limit=50&offset=50"
```

---

### `GET /signals/{id}`

Retrieve a single signal by its ID.

**Response:**

```json
{
  "id": 1,
  "frequency_hz": 8419921066.0,
  "drift_rate": 0.3928,
  "snr": 245.7,
  "rfi_score": 0.05,
  "classification": "narrowband_drifting",
  "confidence": 0.982,
  "is_candidate": true,
  "is_verified": false,
  "observation_id": 1,
  "detected_at": "2025-01-15T08:00:32",
  "image_path": null,
  "ra": 286.86,
  "dec": 12.17,
  "bandwidth_hz": 2.8,
  "notes": null
}
```

**Example:**

```bash
curl http://localhost:8000/signals/1
```

---

### `PATCH /signals/{id}`

Update a signal's classification, verification status, or notes. Used by researchers to manually review and annotate signals.

**Request Body:**

```json
{
  "classification": "candidate_et",
  "is_verified": true,
  "notes": "Confirmed drifting signal at Voyager-1 carrier frequency"
}
```

All fields are optional — only provided fields are updated.

**Response:**

Returns the updated signal object.

**Example:**

```bash
curl -X PATCH http://localhost:8000/signals/1 \
    -H "Content-Type: application/json" \
    -d '{"is_verified": true, "notes": "Confirmed Voyager-1 carrier"}'
```

---

### `GET /candidates`

List signals promoted to ET candidate status, ordered by SNR (descending). This is a convenience endpoint equivalent to `GET /signals?is_candidate=true`.

**Response:**

```json
{
  "candidates": [
    {
      "id": 1,
      "signal_id": 1,
      "observation_id": 1,
      "frequency_hz": 8419921066.0,
      "drift_rate": 0.3928,
      "snr": 245.7,
      "classification": "narrowband_drifting",
      "confidence": 0.982,
      "catalog_matches": [
        {
          "catalog": "SIMBAD",
          "source_name": "Voyager 1",
          "distance_arcmin": 0.02
        }
      ],
      "astrolens_match": null,
      "notes": null,
      "created_at": "2025-01-15T08:00:32"
    }
  ],
  "total": 1
}
```

**Example:**

```bash
curl http://localhost:8000/candidates
```

---

### `GET /stats`

Aggregate processing statistics across all observations.

**Response:**

```json
{
  "total_observations": 48,
  "total_signals": 89432,
  "total_candidates": 15,
  "total_rfi_rejected": 87210,
  "avg_processing_time_ms": 320,
  "avg_signals_per_observation": 1863,
  "candidate_rate": 0.017,
  "rfi_rate": 0.975,
  "classifications": {
    "narrowband_drifting": 142,
    "narrowband_stationary": 1823,
    "broadband": 234,
    "pulsed": 12,
    "chirp": 3,
    "rfi_terrestrial": 45210,
    "rfi_satellite": 42000,
    "noise": 8,
    "candidate_et": 0
  }
}
```

**Example:**

```bash
curl http://localhost:8000/stats
```

---

### `GET /catalog/crossref`

Cross-reference sky coordinates against SIMBAD, NVSS, FIRST, and ATNF Pulsar catalogs. Results are cached for 24 hours.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ra` | float | Yes | Right ascension in degrees |
| `dec` | float | Yes | Declination in degrees |
| `radius_arcmin` | float | No (default: 5.0) | Search radius in arcminutes |

**Response:**

```json
{
  "ra": 286.86,
  "dec": 12.17,
  "radius_arcmin": 5.0,
  "matches": [
    {
      "catalog": "SIMBAD",
      "source_name": "Voyager 1",
      "ra": 286.862,
      "dec": 12.171,
      "distance_arcmin": 0.02,
      "flux_density": null,
      "spectral_type": "Spacecraft",
      "notes": "NASA deep space probe"
    },
    {
      "catalog": "NVSS",
      "source_name": "NVSS J190726+121015",
      "ra": 286.86,
      "dec": 12.17,
      "distance_arcmin": 1.3,
      "flux_density": 12.4,
      "spectral_type": null,
      "notes": "1.4 GHz continuum source"
    }
  ],
  "is_known_source": true,
  "closest_match": {
    "catalog": "SIMBAD",
    "source_name": "Voyager 1",
    "distance_arcmin": 0.02
  }
}
```

**Example:**

```bash
curl "http://localhost:8000/catalog/crossref?ra=286.86&dec=12.17&radius_arcmin=5"
```

---

### `GET /astrolens/crossref`

Check for AstroLens optical anomalies near a radio position. Requires AstroLens artifacts to be available.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ra` | float | Yes | Right ascension in degrees |
| `dec` | float | Yes | Declination in degrees |
| `radius_arcmin` | float | No (default: 2.0) | Search radius in arcminutes |

**Response:**

```json
{
  "ra": 286.86,
  "dec": 12.17,
  "radius_arcmin": 2.0,
  "anomalies_found": 0,
  "matches": []
}
```

**Example:**

```bash
curl "http://localhost:8000/astrolens/crossref?ra=286.86&dec=12.17"
```

---

### `WS /ws/live`

WebSocket endpoint for real-time signal updates during processing. Clients receive JSON messages as signals are detected and classified.

**Message Types:**

```json
{
  "type": "signal_detected",
  "data": {
    "frequency_hz": 8419921066.0,
    "drift_rate": 0.3928,
    "snr": 245.7,
    "classification": "narrowband_drifting",
    "confidence": 0.982
  }
}
```

```json
{
  "type": "processing_complete",
  "data": {
    "file": "voyager1.h5",
    "total_signals": 847,
    "candidates": 1,
    "processing_time_ms": 60
  }
}
```

```json
{
  "type": "candidate_found",
  "data": {
    "id": 1,
    "frequency_hz": 8419921066.0,
    "drift_rate": 0.3928,
    "snr": 245.7,
    "classification": "narrowband_drifting"
  }
}
```

**Example (Python):**

```python
import asyncio
import websockets
import json

async def listen():
    async with websockets.connect("ws://localhost:8000/ws/live") as ws:
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "candidate_found":
                print(f"Candidate: {data['data']['frequency_hz']} Hz, "
                      f"SNR {data['data']['snr']}")

asyncio.run(listen())
```

**Example (JavaScript):**

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/live");

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "candidate_found") {
        console.log(`Candidate: ${data.data.frequency_hz} Hz, SNR ${data.data.snr}`);
    }
};
```

---

## Error Responses

All endpoints return standard HTTP error codes with JSON bodies:

```json
{
  "detail": "Signal with id 999 not found"
}
```

| Code | Meaning |
|------|---------|
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 422 | Validation error (missing required fields) |
| 500 | Internal server error |

---

## CORS

CORS is enabled for all origins by default, allowing browser-based clients to connect from any domain. Configure allowed origins in `api/main.py` for production deployments.
