"""Tests for privacy controls (retention, deletion, audit log)."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.database import init_database
from src.database.face_db import embedding_to_bytes
from src.utils.privacy import AuditLogger, PrivacyManager


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Provide an initialized temporary database."""
    path = str(tmp_path / "test.db")
    init_database(path)
    return path


@pytest.fixture
def populated_db(db_path: str) -> str:
    """Database with test persons and attendance."""
    conn = sqlite3.connect(db_path)
    rng = np.random.default_rng(0)

    conn.execute("INSERT INTO persons (id, name) VALUES (1, 'Alice')")
    conn.execute("INSERT INTO persons (id, name) VALUES (2, 'Bob')")

    for pid in [1, 2]:
        emb = rng.standard_normal(512).astype(np.float32)
        conn.execute(
            "INSERT INTO embeddings (person_id, embedding, model_version) VALUES (?, ?, ?)",
            (pid, embedding_to_bytes(emb), "v1"),
        )

    today = datetime.now().strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO attendance (person_id, confidence, status, timestamp) VALUES (?, ?, ?, ?)",
        (1, 0.95, "present", f"{today} 08:50:00"),
    )
    conn.execute(
        "INSERT INTO attendance (person_id, confidence, status, timestamp) VALUES (?, ?, ?, ?)",
        (2, 0.90, "present", f"{today} 09:00:00"),
    )

    conn.commit()
    conn.close()
    return db_path


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_log_creates_entry(self, db_path: str) -> None:
        """Test audit log creates an entry."""
        audit = AuditLogger(db_path)
        entry_id = audit.log("enroll", person_id=1, details="Test enrollment")

        assert entry_id > 0

    def test_log_with_ip(self, db_path: str) -> None:
        """Test audit log records IP address."""
        audit = AuditLogger(db_path)
        audit.log("verify", person_id=1, ip_address="192.168.1.1")

        entries = audit.get_audit_log(person_id=1)
        assert len(entries) == 1
        assert entries[0]["ip_address"] == "192.168.1.1"

    def test_get_audit_log_filter_person(self, db_path: str) -> None:
        """Test filtering audit log by person ID."""
        audit = AuditLogger(db_path)
        audit.log("enroll", person_id=1)
        audit.log("verify", person_id=2)

        entries = audit.get_audit_log(person_id=1)
        assert len(entries) == 1
        assert entries[0]["person_id"] == 1

    def test_get_audit_log_filter_action(self, db_path: str) -> None:
        """Test filtering audit log by action type."""
        audit = AuditLogger(db_path)
        audit.log("enroll", person_id=1)
        audit.log("verify", person_id=1)
        audit.log("delete", person_id=2)

        entries = audit.get_audit_log(action="enroll")
        assert len(entries) == 1

    def test_get_audit_log_ordering(self, db_path: str) -> None:
        """Test audit log entries are ordered by ID descending (most recent first)."""
        audit = AuditLogger(db_path)
        audit.log("first")
        audit.log("second")
        audit.log("third")

        entries = audit.get_audit_log()
        # Most recent entry (highest ID) should be first
        assert len(entries) == 3
        assert entries[0]["id"] > entries[1]["id"] > entries[2]["id"]

    def test_get_audit_log_limit(self, db_path: str) -> None:
        """Test audit log respects limit."""
        audit = AuditLogger(db_path)
        for i in range(10):
            audit.log(f"action_{i}")

        entries = audit.get_audit_log(limit=3)
        assert len(entries) == 3


class TestPrivacyManager:
    """Tests for PrivacyManager class."""

    def test_delete_person_completely(self, populated_db: str) -> None:
        """Test complete deletion removes all data."""
        pm = PrivacyManager(populated_db)
        result = pm.delete_person_completely(1, requester_ip="127.0.0.1")

        assert result["success"] is True
        assert result["attendance_records_deleted"] == 1

        # Verify person is gone
        conn = sqlite3.connect(populated_db)
        person = conn.execute("SELECT * FROM persons WHERE id = 1").fetchone()
        assert person is None

        # Verify embeddings are gone
        embs = conn.execute("SELECT * FROM embeddings WHERE person_id = 1").fetchall()
        assert len(embs) == 0

        # Verify attendance is gone
        att = conn.execute("SELECT * FROM attendance WHERE person_id = 1").fetchall()
        assert len(att) == 0

        conn.close()

    def test_delete_anonymizes_audit(self, populated_db: str) -> None:
        """Test deletion anonymizes audit log entries."""
        pm = PrivacyManager(populated_db)
        # Create some audit entries first
        pm.audit.log("verify", person_id=1, details="Verified Alice")
        pm.delete_person_completely(1)

        entries = pm.audit.get_audit_log(person_id=1)
        for entry in entries:
            if entry["action"] != "delete":
                assert entry["details"] == "[DELETED]"

    def test_delete_nonexistent_person(self, populated_db: str) -> None:
        """Test deleting non-existent person returns failure."""
        pm = PrivacyManager(populated_db)
        result = pm.delete_person_completely(999)

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_retention_policy_no_old_data(self, populated_db: str) -> None:
        """Test retention policy with no old data deletes nothing."""
        pm = PrivacyManager(populated_db, retention_days=365)
        result = pm.apply_retention_policy()

        assert result["embeddings_deleted"] == 0
        assert result["attendance_deleted"] == 0
        assert result["audit_deleted"] == 0

    def test_retention_policy_deletes_old_data(self, populated_db: str) -> None:
        """Test retention policy deletes data older than retention period."""
        # Insert old data
        conn = sqlite3.connect(populated_db)
        old_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            "INSERT INTO attendance (person_id, confidence, status, timestamp) VALUES (?, ?, ?, ?)",
            (1, 0.95, "present", old_date),
        )
        conn.execute(
            "INSERT INTO embeddings (person_id, embedding, model_version, created_at) "
            "VALUES (?, ?, ?, ?)",
            (1, b"\x00" * 2048, "v1", old_date),
        )
        conn.commit()
        conn.close()

        pm = PrivacyManager(populated_db, retention_days=365)
        result = pm.apply_retention_policy()

        assert result["embeddings_deleted"] == 1
        assert result["attendance_deleted"] == 1

    def test_export_person_data(self, populated_db: str) -> None:
        """Test data export includes all data types."""
        pm = PrivacyManager(populated_db)
        pm.audit.log("verify", person_id=1)

        data = pm.export_person_data(1)

        assert "person" in data
        assert data["person"]["name"] == "Alice"
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert "attendance" in data
        assert len(data["attendance"]) == 1
        assert "exported_at" in data

    def test_export_nonexistent_person(self, populated_db: str) -> None:
        """Test export for non-existent person returns error."""
        pm = PrivacyManager(populated_db)
        data = pm.export_person_data(999)

        assert "error" in data
