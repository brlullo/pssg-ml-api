import sqlite3
import sys
from datetime import datetime, timezone

from app.config import settings


def _use_postgres() -> bool:
    return bool(settings.DATABASE_URL)


# --- SQLite path (local dev, tests, CI) ---


def _sqlite_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.LOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            response_time_ms REAL,
            status_code INTEGER
        )
    """)
    conn.commit()
    return conn


# --- Postgres path (production; requires DATABASE_URL) ---


def _pg_connection():
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(settings.DATABASE_URL, row_factory=dict_row)


def init_db() -> None:
    """Create the request_logs table if needed. Called once at startup."""
    if not _use_postgres():
        _sqlite_connection().close()
        return
    with _pg_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                model_name TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                response_time_ms REAL,
                status_code INTEGER
            )
        """)


def log_request(
    model_name: str,
    endpoint: str,
    response_time_ms: float,
    status_code: int,
) -> None:
    """Record one API request. Never raises: usage logging must not break predictions."""
    try:
        row = (
            datetime.now(timezone.utc),
            model_name,
            endpoint,
            round(response_time_ms, 2),
            status_code,
        )
        if _use_postgres():
            with _pg_connection() as conn:
                conn.execute(
                    "INSERT INTO request_logs (timestamp, model_name, endpoint, response_time_ms, status_code) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    row,
                )
        else:
            conn = _sqlite_connection()
            try:
                conn.execute(
                    "INSERT INTO request_logs (timestamp, model_name, endpoint, response_time_ms, status_code) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (row[0].isoformat(), *row[1:]),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        print(f"[logging] failed to record request: {e}", file=sys.stderr)


def get_usage_stats() -> dict:
    if _use_postgres():
        with _pg_connection() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM request_logs").fetchone()["n"]
            per_model = {
                r["model_name"]: r["count"]
                for r in conn.execute(
                    "SELECT model_name, COUNT(*) AS count FROM request_logs GROUP BY model_name"
                )
            }
            monthly = [
                {"month": r["month"], "count": r["count"]}
                for r in conn.execute(
                    "SELECT to_char(timestamp, 'YYYY-MM') AS month, COUNT(*) AS count "
                    "FROM request_logs GROUP BY month ORDER BY month"
                )
            ]
    else:
        conn = _sqlite_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
            per_model = {
                r["model_name"]: r["count"]
                for r in conn.execute(
                    "SELECT model_name, COUNT(*) as count FROM request_logs GROUP BY model_name"
                )
            }
            monthly = [
                {"month": r["month"], "count": r["count"]}
                for r in conn.execute(
                    "SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) as count "
                    "FROM request_logs GROUP BY month ORDER BY month"
                )
            ]
        finally:
            conn.close()

    return {
        "total_requests": total,
        "per_model": per_model,
        "monthly_trend": monthly,
    }
