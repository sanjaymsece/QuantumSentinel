"""Shared test configuration.

API tests use a dedicated PostgreSQL database so they can exercise the real
archive without touching the dashboard's database.  Set QS_TEST_DATABASE_URL
to override the default test DSN.
"""

from __future__ import annotations

import os

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

TEST_DATABASE_URL = os.getenv(
    "QS_TEST_DATABASE_URL",
    "postgresql://postgres@127.0.0.1:5432/quantum_sentinel_test",
)

# app.config reads this during collection; override before any app import.
os.environ["QS_DATABASE_URL"] = TEST_DATABASE_URL


def _ensure_test_database() -> bool:
    """Create the test database when possible, without assuming privileges."""
    try:
        parameters = conninfo_to_dict(TEST_DATABASE_URL)
        database = str(parameters.get("dbname") or "")
        if not database or database == "postgres":
            return False
        parameters["dbname"] = "postgres"
        maintenance_dsn = make_conninfo(**parameters)
        with psycopg.connect(
            maintenance_dsn, autocommit=True, connect_timeout=2
        ) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", [database]
            ).fetchone()
            if not exists:
                conn.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
                )
        return True
    except Exception:
        return False


DATABASE_AVAILABLE = _ensure_test_database()
