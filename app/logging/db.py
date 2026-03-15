import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def _get_connection() -> sqlite3.Connection:
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


def log_request(
    model_name: str,
    endpoint: str,
    response_time_ms: float,
    status_code: int,
) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO request_logs (timestamp, model_name, endpoint, response_time_ms, status_code) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                model_name,
                endpoint,
                round(response_time_ms, 2),
                status_code,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_usage_stats() -> dict:
    conn = _get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]

        per_model = {}
        for row in conn.execute(
            "SELECT model_name, COUNT(*) as count FROM request_logs GROUP BY model_name"
        ):
            per_model[row["model_name"]] = row["count"]

        monthly = []
        for row in conn.execute(
            "SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) as count "
            "FROM request_logs GROUP BY month ORDER BY month"
        ):
            monthly.append({"month": row["month"], "count": row["count"]})

        return {
            "total_requests": total,
            "per_model": per_model,
            "monthly_trend": monthly,
        }
    finally:
        conn.close()
