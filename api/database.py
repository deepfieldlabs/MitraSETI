"""
Async SQLite database layer for MitraSETI.

Uses aiosqlite for non-blocking I/O so the FastAPI event-loop is never
blocked during DB operations.  Three tables:

    signals       – every detected narrowband hit
    observations  – each processed filterbank file
    candidates    – signals promoted to ET-candidate status
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import DB_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SQL Schemas
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    frequency_hz    REAL    NOT NULL,
    drift_rate      REAL    NOT NULL DEFAULT 0.0,
    snr             REAL    NOT NULL DEFAULT 0.0,
    rfi_score       REAL    DEFAULT 0.0,
    classification  TEXT    DEFAULT 'unknown',
    confidence      REAL    DEFAULT 0.0,
    is_candidate    INTEGER DEFAULT 0,
    is_verified     INTEGER DEFAULT 0,
    observation_id  INTEGER REFERENCES observations(id),
    detected_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    image_path      TEXT,
    ra              REAL,
    dec             REAL,
    bandwidth_hz    REAL,
    notes           TEXT
);
"""

_CREATE_OBSERVATIONS = """
CREATE TABLE IF NOT EXISTS observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path        TEXT    NOT NULL,
    source_name      TEXT,
    ra               REAL,
    dec              REAL,
    duration         REAL,
    processed_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    total_signals    INTEGER DEFAULT 0,
    candidates_found INTEGER DEFAULT 0,
    status           TEXT    DEFAULT 'pending',
    error_message    TEXT
);
"""

_CREATE_CANDIDATES = """
CREATE TABLE IF NOT EXISTS candidates (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id        INTEGER NOT NULL REFERENCES signals(id),
    observation_id   INTEGER REFERENCES observations(id),
    notes            TEXT,
    catalog_matches  TEXT,
    astrolens_match  TEXT,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_signals_snr ON signals(snr);",
    "CREATE INDEX IF NOT EXISTS idx_signals_candidate ON signals(is_candidate);",
    "CREATE INDEX IF NOT EXISTS idx_signals_observation ON signals(observation_id);",
    "CREATE INDEX IF NOT EXISTS idx_signals_classification ON signals(classification);",
    "CREATE INDEX IF NOT EXISTS idx_observations_status ON observations(status);",
    "CREATE INDEX IF NOT EXISTS idx_candidates_signal ON candidates(signal_id);",
]


# ─────────────────────────────────────────────────────────────────────────────
# SignalDB
# ─────────────────────────────────────────────────────────────────────────────


class SignalDB:
    """Async SQLite database manager for MitraSETI signals."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = str(db_path or DB_PATH)

    # ── lifecycle ────────────────────────────────────────────────────────

    async def init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(_CREATE_SIGNALS)
            await db.execute(_CREATE_OBSERVATIONS)
            await db.execute(_CREATE_CANDIDATES)
            for idx_sql in _INDEXES:
                await db.execute(idx_sql)
            await db.commit()
        logger.info("Database initialized at %s", self.db_path)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(cursor: aiosqlite.Cursor, row: aiosqlite.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict."""
        if row is None:
            return {}
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

    # ── signals CRUD ─────────────────────────────────────────────────────

    async def add_signal(self, signal_data: Dict[str, Any]) -> int:
        """
        Insert a new signal and return its id.

        Expected keys mirror the ``signals`` table columns.
        """
        cols = []
        placeholders = []
        values = []
        allowed = {
            "frequency_hz",
            "drift_rate",
            "snr",
            "rfi_score",
            "classification",
            "confidence",
            "is_candidate",
            "is_verified",
            "observation_id",
            "detected_at",
            "image_path",
            "ra",
            "dec",
            "bandwidth_hz",
            "notes",
        }
        for key, val in signal_data.items():
            if key in allowed:
                cols.append(key)
                placeholders.append("?")
                values.append(val)

        if "detected_at" not in signal_data:
            cols.append("detected_at")
            placeholders.append("?")
            values.append(datetime.now(timezone.utc).isoformat())

        sql = f"INSERT INTO signals ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(sql, values)
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    async def get_signal(self, signal_id: int) -> Dict[str, Any]:
        """Return a single signal by id, or empty dict."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM signals WHERE id = ?", (signal_id,))
            row = await cursor.fetchone()
            if row is None:
                return {}
            return dict(row)

    async def get_signals(
        self,
        *,
        min_snr: Optional[float] = None,
        max_rfi_score: Optional[float] = None,
        min_drift_rate: Optional[float] = None,
        max_drift_rate: Optional[float] = None,
        classification: Optional[str] = None,
        is_candidate: Optional[bool] = None,
        observation_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return signals with optional filtering."""
        clauses: List[str] = []
        params: List[Any] = []

        if min_snr is not None:
            clauses.append("snr >= ?")
            params.append(min_snr)
        if max_rfi_score is not None:
            clauses.append("rfi_score <= ?")
            params.append(max_rfi_score)
        if min_drift_rate is not None:
            clauses.append("drift_rate >= ?")
            params.append(min_drift_rate)
        if max_drift_rate is not None:
            clauses.append("drift_rate <= ?")
            params.append(max_drift_rate)
        if classification is not None:
            clauses.append("classification = ?")
            params.append(classification)
        if is_candidate is not None:
            clauses.append("is_candidate = ?")
            params.append(int(is_candidate))
        if observation_id is not None:
            clauses.append("observation_id = ?")
            params.append(observation_id)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM signals{where} ORDER BY detected_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def update_signal(self, signal_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update signal fields and return the updated record."""
        allowed = {
            "classification",
            "confidence",
            "rfi_score",
            "is_candidate",
            "is_verified",
            "image_path",
            "notes",
        }
        sets: List[str] = []
        values: List[Any] = []
        for key, val in data.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                values.append(val)

        if not sets:
            return await self.get_signal(signal_id)

        values.append(signal_id)
        sql = f"UPDATE signals SET {', '.join(sets)} WHERE id = ?"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(sql, values)
            await db.commit()

        return await self.get_signal(signal_id)

    async def get_candidates(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return only signals marked as ET candidates."""
        sql = "SELECT * FROM signals WHERE is_candidate = 1 ORDER BY snr DESC LIMIT ? OFFSET ?"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql, (limit, offset))
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ── observations ─────────────────────────────────────────────────────

    async def add_observation(self, obs_data: Dict[str, Any]) -> int:
        """Insert an observation record and return its id."""
        cols = []
        placeholders = []
        values = []
        allowed = {
            "file_path",
            "source_name",
            "ra",
            "dec",
            "duration",
            "processed_at",
            "total_signals",
            "candidates_found",
            "status",
            "error_message",
        }
        for key, val in obs_data.items():
            if key in allowed:
                cols.append(key)
                placeholders.append("?")
                values.append(val)

        if "processed_at" not in obs_data:
            cols.append("processed_at")
            placeholders.append("?")
            values.append(datetime.now(timezone.utc).isoformat())

        sql = f"INSERT INTO observations ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(sql, values)
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    async def update_observation(self, obs_id: int, data: Dict[str, Any]) -> None:
        """Update an observation record."""
        allowed = {
            "total_signals",
            "candidates_found",
            "status",
            "error_message",
        }
        sets: List[str] = []
        values: List[Any] = []
        for key, val in data.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                values.append(val)
        if not sets:
            return
        values.append(obs_id)
        sql = f"UPDATE observations SET {', '.join(sets)} WHERE id = ?"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(sql, values)
            await db.commit()

    # ── candidates table ─────────────────────────────────────────────────

    async def add_candidate_record(
        self,
        signal_id: int,
        observation_id: Optional[int] = None,
        notes: str = "",
        catalog_matches: str = "",
        astrolens_match: str = "",
    ) -> int:
        """Promote a signal to the candidates table."""
        sql = (
            "INSERT INTO candidates "
            "(signal_id, observation_id, notes, catalog_matches, astrolens_match) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                sql,
                (signal_id, observation_id, notes, catalog_matches, astrolens_match),
            )
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    # ── statistics ───────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all tables."""
        async with aiosqlite.connect(self.db_path) as db:
            total_signals = (await (await db.execute("SELECT COUNT(*) FROM signals")).fetchone())[0]
            total_candidates = (
                await (
                    await db.execute("SELECT COUNT(*) FROM signals WHERE is_candidate = 1")
                ).fetchone()
            )[0]
            verified = (
                await (
                    await db.execute("SELECT COUNT(*) FROM signals WHERE is_verified = 1")
                ).fetchone()
            )[0]
            total_observations = (
                await (await db.execute("SELECT COUNT(*) FROM observations")).fetchone()
            )[0]
            avg_snr_row = await (
                await db.execute("SELECT AVG(snr) FROM signals WHERE snr > 0")
            ).fetchone()
            avg_snr = round(avg_snr_row[0], 2) if avg_snr_row[0] is not None else 0.0

            return {
                "total_signals": total_signals,
                "total_candidates": total_candidates,
                "verified_signals": verified,
                "total_observations": total_observations,
                "average_snr": avg_snr,
            }
