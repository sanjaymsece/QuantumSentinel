"""PostgreSQL-backed archive for completed BB84 sessions.

The archive is intentionally append-only.  Every successful ``POST /run`` is
written before the response is returned, so a session acknowledged by the API
is always present in the database.  Schema creation is idempotent and happens
on startup; the API stays online in a degraded state when PostgreSQL is
unavailable and resumes writes once it reconnects.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

from psycopg import Error as PsycopgError
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

logger = logging.getLogger("quantumsentinel.database")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS run_sessions (
    session_id          UUID PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    engine              TEXT NOT NULL,
    engine_requested    TEXT NOT NULL,
    eve_mode            TEXT NOT NULL,
    eve_strength        SMALLINT NOT NULL,
    noise_percent       SMALLINT NOT NULL,
    qubits              INTEGER NOT NULL,
    channel             TEXT NOT NULL,
    seed                BIGINT NOT NULL,

    qber                DOUBLE PRECISION NOT NULL,
    threshold           DOUBLE PRECISION NOT NULL,
    standard_detected   BOOLEAN NOT NULL,
    ml_detected         BOOLEAN NOT NULL,
    ml_confidence       DOUBLE PRECISION NOT NULL,
    attack_probability  DOUBLE PRECISION NOT NULL,
    attack_present      BOOLEAN NOT NULL,
    attack_mode         TEXT NOT NULL,

    block_qber          JSONB NOT NULL,
    sifted_bits         INTEGER NOT NULL,
    sifted_rate         DOUBLE PRECISION NOT NULL,
    x_qber              DOUBLE PRECISION NOT NULL,
    z_qber              DOUBLE PRECISION NOT NULL,
    expected_qber       DOUBLE PRECISION NOT NULL,
    elapsed_ms          DOUBLE PRECISION NOT NULL,

    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    model_threshold     DOUBLE PRECISION NOT NULL,
    features            JSONB NOT NULL,

    CONSTRAINT run_sessions_eve_strength_range
        CHECK (eve_strength BETWEEN 1 AND 100),
    CONSTRAINT run_sessions_noise_range
        CHECK (noise_percent BETWEEN 0 AND 20),
    CONSTRAINT run_sessions_qubits_range
        CHECK (qubits BETWEEN 100 AND 5000),
    CONSTRAINT run_sessions_qber_range
        CHECK (qber >= 0 AND qber <= 100),
    CONSTRAINT run_sessions_probability_range
        CHECK (attack_probability >= 0 AND attack_probability <= 1)
);

CREATE INDEX IF NOT EXISTS run_sessions_created_at_idx
    ON run_sessions (created_at DESC, session_id DESC);

CREATE INDEX IF NOT EXISTS run_sessions_ml_created_at_idx
    ON run_sessions (ml_detected, created_at DESC);

CREATE INDEX IF NOT EXISTS run_sessions_attack_created_at_idx
    ON run_sessions (attack_present, created_at DESC);
"""

SUMMARY_COLUMNS = """
session_id, created_at, engine, eve_mode, eve_strength, noise_percent,
qubits, channel, qber, threshold, standard_detected, ml_detected,
ml_confidence, attack_probability, attack_present, attack_mode, elapsed_ms
"""

DETAIL_COLUMNS = """
session_id, created_at, engine, engine_requested, eve_mode, eve_strength,
noise_percent, qubits, channel, seed, qber, threshold, standard_detected,
ml_detected, ml_confidence, attack_probability, attack_present, attack_mode,
block_qber, sifted_bits, sifted_rate, x_qber, z_qber, expected_qber,
elapsed_ms, model_name, model_version, model_threshold, features
"""


class SessionStoreError(RuntimeError):
    """Raised when a session cannot be read from or written to PostgreSQL."""


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """A fully resolved run, ready to be archived."""

    session_id: str
    engine: str
    engine_requested: str
    eve_mode: str
    eve_strength: int
    noise: int
    qubits: int
    channel: str
    seed: int
    qber: float
    threshold: float
    standard_detected: bool
    ml_detected: bool
    ml_confidence: float
    attack_probability: float
    attack_present: bool
    attack_mode: str
    block_qber: list[float]
    sifted_bits: int
    sifted_rate: float
    x_qber: float
    z_qber: float
    expected_qber: float
    elapsed_ms: float
    model_name: str
    model_version: str
    model_threshold: float
    features: dict[str, float]

    def as_parameters(self) -> tuple[Any, ...]:
        values = asdict(self)
        return (
            values["session_id"],
            values["engine"],
            values["engine_requested"],
            values["eve_mode"],
            values["eve_strength"],
            values["noise"],
            values["qubits"],
            values["channel"],
            values["seed"],
            values["qber"],
            values["threshold"],
            values["standard_detected"],
            values["ml_detected"],
            values["ml_confidence"],
            values["attack_probability"],
            values["attack_present"],
            values["attack_mode"],
            Jsonb(values["block_qber"]),
            values["sifted_bits"],
            values["sifted_rate"],
            values["x_qber"],
            values["z_qber"],
            values["expected_qber"],
            values["elapsed_ms"],
            values["model_name"],
            values["model_version"],
            values["model_threshold"],
            Jsonb(values["features"]),
        )


INSERT_SQL = """
INSERT INTO run_sessions (
    session_id, engine, engine_requested, eve_mode, eve_strength,
    noise_percent, qubits, channel, seed, qber, threshold,
    standard_detected, ml_detected, ml_confidence, attack_probability,
    attack_present, attack_mode, block_qber, sifted_bits, sifted_rate,
    x_qber, z_qber, expected_qber, elapsed_ms, model_name, model_version,
    model_threshold, features
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""


class PostgresSessionRepository:
    """Small, thread-safe repository around a psycopg connection pool."""

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout: float = 5.0,
    ) -> None:
        if not dsn:
            raise SessionStoreError("QS_DATABASE_URL is empty")
        self.dsn = dsn
        self._connect_timeout = connect_timeout
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=connect_timeout,
            kwargs={"row_factory": dict_row},
            open=False,
            name="quantumsentinel-sessions",
        )

    @property
    def database_name(self) -> str:
        try:
            return str(conninfo_to_dict(self.dsn).get("dbname") or "")
        except Exception:  # pragma: no cover - malformed DSN is reported elsewhere
            return ""

    def initialize(self) -> None:
        """Open the pool and create the archive schema when necessary."""
        try:
            if self._pool.closed:
                self._pool.open(wait=True, timeout=self._connect_timeout)
            self._ensure_schema()
        except Exception as exc:
            raise SessionStoreError(
                f"PostgreSQL archive is unavailable ({exc.__class__.__name__}: {exc})"
            ) from exc

    def close(self) -> None:
        self._pool.close()

    def ping(self) -> bool:
        """Return True when a database round trip succeeds."""
        try:
            if self._pool.closed:
                self._pool.open(wait=False)
            with self._pool.connection(timeout=self._connect_timeout) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def save(self, record: SessionRecord) -> None:
        self._ensure_schema()
        try:
            with self._pool.connection(timeout=self._connect_timeout) as conn:
                with conn.cursor() as cur:
                    cur.execute(INSERT_SQL, record.as_parameters())
        except PsycopgError as exc:
            raise SessionStoreError(f"Could not archive session: {exc}") from exc

    def list_sessions(
        self,
        *,
        limit: int,
        offset: int,
        ml_detected: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        self._ensure_schema()
        where = ""
        parameters: list[Any] = []
        if ml_detected is not None:
            where = "WHERE ml_detected = %s"
            parameters.append(ml_detected)

        count_sql = f"SELECT count(*) AS total FROM run_sessions {where}"
        list_sql = f"""
            SELECT {SUMMARY_COLUMNS}
            FROM run_sessions
            {where}
            ORDER BY created_at DESC, session_id DESC
            LIMIT %s OFFSET %s
        """
        try:
            with self._pool.connection(timeout=self._connect_timeout) as conn:
                with conn.cursor() as cur:
                    cur.execute(count_sql, parameters)
                    total = int(cur.fetchone()["total"])
                    cur.execute(list_sql, [*parameters, limit, offset])
                    rows = cur.fetchall()
            return [dict(row) for row in rows], total
        except PsycopgError as exc:
            raise SessionStoreError(f"Could not read session archive: {exc}") from exc

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        self._ensure_schema()
        query = f"SELECT {DETAIL_COLUMNS} FROM run_sessions WHERE session_id = %s"
        try:
            with self._pool.connection(timeout=self._connect_timeout) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, [session_id])
                    row = cur.fetchone()
            return dict(row) if row is not None else None
        except PsycopgError as exc:
            raise SessionStoreError(f"Could not read session: {exc}") from exc

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            try:
                if self._pool.closed:
                    self._pool.open(wait=True, timeout=self._connect_timeout)
                with self._pool.connection(timeout=self._connect_timeout) as conn:
                    with conn.cursor() as cur:
                        cur.execute(SCHEMA_SQL)
                self._schema_ready = True
                logger.info(
                    "Session archive schema ready in database %s",
                    self.database_name or "(unknown)",
                )
            except Exception as exc:
                raise SessionStoreError(
                    "PostgreSQL archive is unavailable "
                    f"({exc.__class__.__name__}: {exc})"
                ) from exc


__all__ = [
    "PostgresSessionRepository",
    "SessionRecord",
    "SessionStoreError",
]
