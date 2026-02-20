"""Tests for database initialization and schema."""

import sqlite3
from pathlib import Path

import pytest

from src.database import init_database


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Provide a temporary database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def initialized_db(db_path: str) -> str:
    """Provide an initialized database."""
    init_database(db_path)
    return db_path


def _get_tables(db_path: str) -> list[str]:
    """Helper to get all table names in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def _get_indexes(db_path: str) -> list[str]:
    """Helper to get all index names in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
    indexes = [row[0] for row in cursor.fetchall()]
    conn.close()
    return indexes


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_creates_all_tables(self, initialized_db: str) -> None:
        """Test that all required tables are created."""
        tables = _get_tables(initialized_db)
        expected = ["attendance", "audit_log", "embeddings", "persons"]
        for table in expected:
            assert table in tables, f"Missing table: {table}"

    def test_creates_indexes(self, initialized_db: str) -> None:
        """Test that performance indexes are created."""
        indexes = _get_indexes(initialized_db)
        expected = [
            "idx_attendance_person_time",
            "idx_embeddings_person",
            "idx_audit_person",
            "idx_attendance_date",
        ]
        for index in expected:
            assert index in indexes, f"Missing index: {index}"

    def test_idempotent_initialization(self, db_path: str) -> None:
        """Test schema can be applied multiple times safely."""
        init_database(db_path)
        init_database(db_path)
        tables = _get_tables(db_path)
        assert "persons" in tables

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that missing parent directories are created."""
        nested_path = str(tmp_path / "nested" / "dir" / "test.db")
        init_database(nested_path)
        assert Path(nested_path).exists()

    def test_persons_table_schema(self, initialized_db: str) -> None:
        """Test persons table has correct columns."""
        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("PRAGMA table_info(persons)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "name" in columns
        assert "created_at" in columns
        assert "updated_at" in columns
        assert "is_active" in columns

    def test_embeddings_table_schema(self, initialized_db: str) -> None:
        """Test embeddings table has correct columns."""
        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("PRAGMA table_info(embeddings)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "person_id" in columns
        assert "embedding" in columns
        assert "model_version" in columns

    def test_attendance_table_schema(self, initialized_db: str) -> None:
        """Test attendance table has correct columns."""
        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("PRAGMA table_info(attendance)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "person_id" in columns
        assert "confidence" in columns
        assert "liveness_score" in columns
        assert "status" in columns

    def test_audit_log_table_schema(self, initialized_db: str) -> None:
        """Test audit_log table has correct columns."""
        conn = sqlite3.connect(initialized_db)
        cursor = conn.execute("PRAGMA table_info(audit_log)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "action" in columns
        assert "person_id" in columns
        assert "details" in columns
        assert "ip_address" in columns


class TestDatabaseConstraints:
    """Tests for database constraints and foreign keys."""

    def test_foreign_key_cascade_delete(self, initialized_db: str) -> None:
        """Test that deleting a person cascades to embeddings."""
        conn = sqlite3.connect(initialized_db)
        conn.execute("PRAGMA foreign_keys = ON")

        # Insert person
        conn.execute("INSERT INTO persons (name) VALUES ('Test Person')")
        person_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Insert embedding
        conn.execute(
            "INSERT INTO embeddings (person_id, embedding, model_version) VALUES (?, ?, ?)",
            (person_id, b"\x00" * 2048, "v1"),
        )
        conn.commit()

        # Verify embedding exists
        count = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
        ).fetchone()[0]
        assert count == 1

        # Delete person
        conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        conn.commit()

        # Verify embedding was cascaded
        count = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
        ).fetchone()[0]
        assert count == 0

        conn.close()

    def test_insert_and_retrieve_data(self, initialized_db: str) -> None:
        """Test basic insert and retrieve operations."""
        conn = sqlite3.connect(initialized_db)

        conn.execute("INSERT INTO persons (name) VALUES ('Alice')")
        conn.commit()

        row = conn.execute("SELECT name FROM persons WHERE name = 'Alice'").fetchone()
        assert row[0] == "Alice"

        conn.close()

    def test_attendance_references_person(self, initialized_db: str) -> None:
        """Test attendance record references a valid person."""
        conn = sqlite3.connect(initialized_db)

        conn.execute("INSERT INTO persons (name) VALUES ('Bob')")
        person_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO attendance (person_id, confidence, status) VALUES (?, ?, ?)",
            (person_id, 0.95, "present"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT confidence, status FROM attendance WHERE person_id = ?",
            (person_id,),
        ).fetchone()
        assert row[0] == 0.95
        assert row[1] == "present"

        conn.close()
