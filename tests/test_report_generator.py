"""Tests for report generation."""

import csv
import io
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.database import init_database
from src.database.attendance_db import AttendanceDatabase
from src.database.face_db import FaceDatabase, embedding_to_bytes
from src.reporting.report_generator import ReportGenerator


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Provide an initialized temporary database."""
    path = str(tmp_path / "test.db")
    init_database(path)
    return path


@pytest.fixture
def populated_db(db_path: str) -> str:
    """Database with persons and attendance records."""
    conn = sqlite3.connect(db_path)
    rng = np.random.default_rng(0)

    conn.execute("INSERT INTO persons (id, name) VALUES (1, 'Alice')")
    conn.execute("INSERT INTO persons (id, name) VALUES (2, 'Bob')")
    conn.execute("INSERT INTO persons (id, name) VALUES (3, 'Carol')")

    for pid in [1, 2, 3]:
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
        (2, 0.90, "late", f"{today} 09:30:00"),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def report_gen(populated_db: str) -> ReportGenerator:
    """Provide a ReportGenerator with populated data."""
    att_db = AttendanceDatabase(populated_db)
    face_db = FaceDatabase(populated_db)
    return ReportGenerator(att_db, face_db)


class TestDailyCSV:
    """Tests for daily CSV report generation."""

    def test_valid_csv_format(self, report_gen: ReportGenerator) -> None:
        """Test that generated CSV is parseable."""
        csv_content = report_gen.generate_daily_csv(datetime.now())
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)
        assert len(rows) > 0

    def test_contains_present_person(self, report_gen: ReportGenerator) -> None:
        """Test CSV contains present person."""
        csv_content = report_gen.generate_daily_csv(datetime.now())
        assert "Alice" in csv_content
        assert "Present" in csv_content

    def test_contains_late_person(self, report_gen: ReportGenerator) -> None:
        """Test CSV contains late person."""
        csv_content = report_gen.generate_daily_csv(datetime.now())
        assert "Bob" in csv_content
        assert "Late" in csv_content

    def test_contains_absent_person(self, report_gen: ReportGenerator) -> None:
        """Test CSV contains absent person."""
        csv_content = report_gen.generate_daily_csv(datetime.now())
        assert "Carol" in csv_content
        assert "Absent" in csv_content

    def test_contains_summary(self, report_gen: ReportGenerator) -> None:
        """Test CSV contains summary section."""
        csv_content = report_gen.generate_daily_csv(datetime.now())
        assert "Summary" in csv_content
        assert "Attendance Rate" in csv_content


class TestDailyMarkdown:
    """Tests for daily Markdown report generation."""

    def test_valid_markdown(self, report_gen: ReportGenerator) -> None:
        """Test Markdown contains headers and table."""
        md = report_gen.generate_daily_markdown(datetime.now())
        assert "# Attendance Report:" in md
        assert "| Name | Status | Check-in Time |" in md

    def test_contains_all_persons(self, report_gen: ReportGenerator) -> None:
        """Test Markdown includes all persons."""
        md = report_gen.generate_daily_markdown(datetime.now())
        assert "Alice" in md
        assert "Bob" in md
        assert "Carol" in md


class TestWeeklyCSV:
    """Tests for weekly CSV report generation."""

    def test_weekly_csv_header(self, report_gen: ReportGenerator) -> None:
        """Test weekly CSV has date column headers."""
        today = datetime.now()
        csv_content = report_gen.generate_weekly_csv(today - timedelta(days=6), today)
        assert "Weekly Attendance Report" in csv_content
        assert "Total Days" in csv_content

    def test_weekly_csv_persons(self, report_gen: ReportGenerator) -> None:
        """Test weekly CSV contains all persons."""
        today = datetime.now()
        csv_content = report_gen.generate_weekly_csv(today, today)
        assert "Alice" in csv_content
        assert "Bob" in csv_content

    def test_weekly_csv_default_end_date(self, report_gen: ReportGenerator) -> None:
        """Test weekly CSV with default end date (start + 6 days)."""
        today = datetime.now()
        csv_content = report_gen.generate_weekly_csv(today)
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)
        assert len(rows) > 3


class TestWeeklyMarkdown:
    """Tests for weekly Markdown report generation."""

    def test_weekly_markdown_format(self, report_gen: ReportGenerator) -> None:
        """Test weekly Markdown has attendance matrix."""
        today = datetime.now()
        md = report_gen.generate_weekly_markdown(today, today)
        assert "# Weekly Attendance Report" in md
        assert "## Attendance Matrix" in md


class TestSaveReport:
    """Tests for report file saving."""

    def test_save_creates_file(self, report_gen: ReportGenerator, tmp_path: Path) -> None:
        """Test report is saved to disk."""
        content = report_gen.generate_daily_csv(datetime.now())
        filepath = tmp_path / "reports" / "daily.csv"
        report_gen.save_report(content, filepath)

        assert filepath.exists()
        saved = filepath.read_text()
        assert saved.replace("\r\n", "\n") == content.replace("\r\n", "\n")

    def test_save_creates_parent_dirs(
        self,
        report_gen: ReportGenerator,
        tmp_path: Path,
    ) -> None:
        """Test save creates missing parent directories."""
        filepath = tmp_path / "nested" / "dir" / "report.md"
        report_gen.save_report("test content", filepath)
        assert filepath.exists()


class TestNoData:
    """Tests for reports with no attendance data."""

    def test_daily_csv_empty(self, db_path: str) -> None:
        """Test daily CSV with no attendance records."""
        att_db = AttendanceDatabase(db_path)
        face_db = FaceDatabase(db_path)
        gen = ReportGenerator(att_db, face_db)

        csv_content = gen.generate_daily_csv(datetime.now())
        assert "Summary" in csv_content
        assert "0" in csv_content

    def test_daily_markdown_empty(self, db_path: str) -> None:
        """Test daily Markdown with no attendance records."""
        att_db = AttendanceDatabase(db_path)
        face_db = FaceDatabase(db_path)
        gen = ReportGenerator(att_db, face_db)

        md = gen.generate_daily_markdown(datetime.now())
        assert "0" in md
