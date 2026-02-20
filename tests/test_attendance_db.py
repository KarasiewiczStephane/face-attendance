"""Tests for attendance logging with deduplication."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.database import init_database
from src.database.attendance_db import AttendanceDatabase
from src.database.face_db import embedding_to_bytes


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Provide an initialized temporary database."""
    path = str(tmp_path / "test.db")
    init_database(path)
    return path


@pytest.fixture
def attendance_db(db_path: str) -> AttendanceDatabase:
    """Provide an AttendanceDatabase instance."""
    return AttendanceDatabase(db_path, dedup_hours=4.0)


@pytest.fixture
def populated_db(db_path: str) -> str:
    """Database with two registered persons."""
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO persons (id, name) VALUES (1, 'Alice')")
    conn.execute("INSERT INTO persons (id, name) VALUES (2, 'Bob')")
    rng = np.random.default_rng(0)
    for pid in [1, 2]:
        emb = rng.standard_normal(512).astype(np.float32)
        conn.execute(
            "INSERT INTO embeddings (person_id, embedding, model_version) VALUES (?, ?, ?)",
            (pid, embedding_to_bytes(emb), "v1"),
        )
    conn.commit()
    conn.close()
    return db_path


class TestAttendanceLogging:
    """Tests for attendance log_attendance."""

    def test_log_attendance_creates_record(
        self,
        populated_db: str,
    ) -> None:
        """Test logging attendance creates a record."""
        db = AttendanceDatabase(populated_db)
        att_id = db.log_attendance(1, 0.95, 0.8, "present")

        assert att_id is not None
        assert att_id > 0

    def test_deduplication_within_window(
        self,
        populated_db: str,
    ) -> None:
        """Test duplicate attendance within window returns None."""
        db = AttendanceDatabase(populated_db, dedup_hours=4.0)

        att_id1 = db.log_attendance(1, 0.95)
        assert att_id1 is not None

        att_id2 = db.log_attendance(1, 0.93)
        assert att_id2 is None

    def test_different_persons_not_deduplicated(
        self,
        populated_db: str,
    ) -> None:
        """Test different persons are logged independently."""
        db = AttendanceDatabase(populated_db)

        att1 = db.log_attendance(1, 0.95)
        att2 = db.log_attendance(2, 0.90)

        assert att1 is not None
        assert att2 is not None

    def test_dedup_after_window_expires(
        self,
        populated_db: str,
    ) -> None:
        """Test attendance is allowed after dedup window expires."""
        db = AttendanceDatabase(populated_db, dedup_hours=0.001)  # ~3.6 seconds

        att1 = db.log_attendance(1, 0.95)
        assert att1 is not None

        # Manually insert a record in the past
        conn = sqlite3.connect(populated_db)
        old_time = (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("DELETE FROM attendance WHERE person_id = 1")
        conn.execute(
            "INSERT INTO attendance (person_id, confidence, status, timestamp) VALUES (?, ?, ?, ?)",
            (1, 0.95, "present", old_time),
        )
        conn.commit()
        conn.close()

        att2 = db.log_attendance(1, 0.93)
        assert att2 is not None


class TestDetermineStatus:
    """Tests for attendance status determination."""

    def test_on_time(self) -> None:
        """Test on-time arrival returns 'present'."""
        db = AttendanceDatabase(":memory:")
        timestamp = datetime(2026, 1, 15, 8, 45)
        status = db.determine_status(timestamp, work_start="09:00", late_threshold_minutes=15)
        assert status == "present"

    def test_within_grace_period(self) -> None:
        """Test arrival within grace period returns 'present'."""
        db = AttendanceDatabase(":memory:")
        timestamp = datetime(2026, 1, 15, 9, 10)
        status = db.determine_status(timestamp, work_start="09:00", late_threshold_minutes=15)
        assert status == "present"

    def test_late(self) -> None:
        """Test late arrival returns 'late'."""
        db = AttendanceDatabase(":memory:")
        timestamp = datetime(2026, 1, 15, 9, 30)
        status = db.determine_status(timestamp, work_start="09:00", late_threshold_minutes=15)
        assert status == "late"

    def test_exactly_at_cutoff(self) -> None:
        """Test arrival exactly at grace period cutoff returns 'present'."""
        db = AttendanceDatabase(":memory:")
        timestamp = datetime(2026, 1, 15, 9, 15)
        status = db.determine_status(timestamp, work_start="09:00", late_threshold_minutes=15)
        assert status == "present"


class TestAttendanceQueries:
    """Tests for attendance query methods."""

    def test_get_attendance_by_date(self, populated_db: str) -> None:
        """Test getting attendance records for a date."""
        db = AttendanceDatabase(populated_db)
        db.log_attendance(1, 0.95, 0.8, "present")

        records = db.get_attendance_by_date(datetime.now())
        assert len(records) == 1
        assert records[0]["name"] == "Alice"

    def test_get_attendance_by_date_empty(self, populated_db: str) -> None:
        """Test getting attendance for date with no records."""
        db = AttendanceDatabase(populated_db)
        records = db.get_attendance_by_date(datetime(2020, 1, 1))
        assert len(records) == 0

    def test_get_attendance_range(self, populated_db: str) -> None:
        """Test getting attendance for a date range."""
        db = AttendanceDatabase(populated_db)
        db.log_attendance(1, 0.95)
        db.log_attendance(2, 0.90)

        today = datetime.now()
        records = db.get_attendance_range(today - timedelta(days=1), today + timedelta(days=1))
        assert len(records) == 2

    def test_get_daily_summary(self, populated_db: str) -> None:
        """Test daily summary with present and absent persons."""
        db = AttendanceDatabase(populated_db)
        db.log_attendance(1, 0.95, status="present")

        summary = db.get_daily_summary(datetime.now())

        assert summary["summary"]["total"] == 2
        assert summary["summary"]["present"] == 1
        assert summary["summary"]["absent"] == 1
        assert len(summary["present"]) == 1
        assert len(summary["absent"]) == 1

    def test_get_daily_summary_all_absent(self, populated_db: str) -> None:
        """Test daily summary when no one attended."""
        db = AttendanceDatabase(populated_db)
        summary = db.get_daily_summary(datetime.now())

        assert summary["summary"]["present"] == 0
        assert summary["summary"]["absent"] == 2

    def test_get_daily_summary_late(self, populated_db: str) -> None:
        """Test daily summary correctly categorizes late arrivals."""
        db = AttendanceDatabase(populated_db)

        conn = sqlite3.connect(populated_db)
        conn.execute(
            "INSERT INTO attendance (person_id, confidence, status) VALUES (?, ?, ?)",
            (1, 0.95, "late"),
        )
        conn.commit()
        conn.close()

        summary = db.get_daily_summary(datetime.now())

        assert summary["summary"]["late"] == 1
        assert summary["summary"]["absent"] == 1
