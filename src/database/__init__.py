"""Database initialization and schema management."""

import sqlite3
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def init_database(db_path: str) -> None:
    """Initialize SQLite database with schema.

    Creates all tables, indexes, and enables foreign key constraints.

    Args:
        db_path: Path to the SQLite database file.
    """
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.close()
    logger.info("Database initialized at %s", db_path)
