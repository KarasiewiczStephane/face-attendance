"""Tests for the face attendance dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    EMPLOYEE_NAMES,
    generate_attendance_log,
    generate_face_gallery,
)


class TestGenerateAttendanceLog:
    """Tests for generate_attendance_log."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with expected columns."""
        df = generate_attendance_log()
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "id",
            "person_id",
            "name",
            "timestamp",
            "date",
            "check_in_time",
            "confidence",
            "liveness_score",
            "status",
        }
        assert expected_cols == set(df.columns)

    def test_not_empty(self):
        """Generates a non-empty log."""
        df = generate_attendance_log()
        assert len(df) > 0

    def test_valid_status_values(self):
        """Status column only contains valid values."""
        df = generate_attendance_log()
        valid_statuses = {"present", "late"}
        assert set(df["status"]).issubset(valid_statuses)

    def test_confidence_range(self):
        """Confidence values are between 0 and 1."""
        df = generate_attendance_log()
        assert (df["confidence"] >= 0).all()
        assert (df["confidence"] <= 1).all()

    def test_liveness_range(self):
        """Liveness scores are between 0 and 1."""
        df = generate_attendance_log()
        assert (df["liveness_score"] >= 0).all()
        assert (df["liveness_score"] <= 1).all()

    def test_names_from_employee_list(self):
        """All names come from the EMPLOYEE_NAMES list."""
        df = generate_attendance_log()
        assert set(df["name"]).issubset(set(EMPLOYEE_NAMES))

    def test_no_weekends(self):
        """No attendance records on weekends."""
        df = generate_attendance_log()
        dates = pd.to_datetime(df["date"])
        weekdays = dates.dt.weekday
        assert (weekdays < 5).all()

    def test_unique_ids(self):
        """All record IDs are unique."""
        df = generate_attendance_log()
        assert df["id"].is_unique

    def test_reproducible_with_seed(self):
        """Same seed produces identical output."""
        df1 = generate_attendance_log(seed=99)
        df2 = generate_attendance_log(seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds produce different outputs."""
        df1 = generate_attendance_log(seed=1)
        df2 = generate_attendance_log(seed=2)
        assert not df1.equals(df2)


class TestGenerateFaceGallery:
    """Tests for generate_face_gallery."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with expected columns."""
        df = generate_face_gallery()
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "person_id",
            "name",
            "enrolled_date",
            "embedding_count",
            "last_seen",
            "is_active",
        }
        assert expected_cols == set(df.columns)

    def test_employee_count(self):
        """Contains all employees."""
        df = generate_face_gallery()
        assert len(df) == len(EMPLOYEE_NAMES)

    def test_all_active(self):
        """All employees are active in demo data."""
        df = generate_face_gallery()
        assert df["is_active"].all()

    def test_embedding_count_positive(self):
        """All embedding counts are positive."""
        df = generate_face_gallery()
        assert (df["embedding_count"] > 0).all()

    def test_unique_person_ids(self):
        """Person IDs are unique."""
        df = generate_face_gallery()
        assert df["person_id"].is_unique
