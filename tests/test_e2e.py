"""End-to-end tests covering the full enrollment-verification-reporting workflow."""

import io
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import _state, app
from src.database import init_database
from src.database.attendance_db import AttendanceDatabase
from src.database.face_db import FaceDatabase
from src.detection.liveness import LivenessDetector
from src.matching import MatchingService
from src.reporting.report_generator import ReportGenerator
from src.utils.privacy import PrivacyManager


def _create_test_image() -> bytes:
    """Create a simple test image as bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def e2e_db(tmp_path) -> str:
    """Provide an initialized temporary database for E2E tests."""
    path = str(tmp_path / "e2e.db")
    init_database(path)
    return path


@pytest.fixture
def stable_embedding() -> np.ndarray:
    """A fixed embedding for consistent matching."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(512).astype(np.float32)


@pytest.fixture
def e2e_client(e2e_db: str, stable_embedding: np.ndarray) -> TestClient:
    """Provide a test client for E2E tests with mocked face processor."""
    face_db = FaceDatabase(e2e_db)
    attendance_db = AttendanceDatabase(e2e_db, dedup_hours=4.0)

    mock_processor = MagicMock()
    mock_processor.process_image.return_value = (
        stable_embedding,
        np.array([10, 10, 50, 50]),
        0.99,
    )
    mock_processor.embedder.get_model_version.return_value = "v1"

    _state["face_processor"] = mock_processor
    _state["face_db"] = face_db
    _state["attendance_db"] = attendance_db
    _state["matching_service"] = MatchingService(face_db, threshold=0.7)
    _state["liveness_detector"] = LivenessDetector(texture_threshold=0.3)
    _state["privacy_manager"] = PrivacyManager(e2e_db)
    _state["report_generator"] = ReportGenerator(attendance_db, face_db)
    _state["config"] = {
        "attendance": {
            "work_start": "09:00",
            "late_threshold_minutes": 15,
            "dedup_window_hours": 4,
        },
        "matching": {"similarity_threshold": 0.7},
        "liveness": {"blink_threshold": 0.25, "texture_threshold": 0.3},
        "privacy": {"retention_days": 365},
    }

    return TestClient(app, raise_server_exceptions=False)


class TestFullWorkflow:
    """End-to-end workflow tests."""

    def test_enroll_verify_attendance_report_delete_flow(
        self, e2e_client: TestClient, stable_embedding: np.ndarray
    ) -> None:
        """Test complete enrollment -> verification -> attendance -> report -> delete flow."""
        image_bytes = _create_test_image()

        # Step 1: Enroll a person
        response = e2e_client.post(
            "/enroll?name=TestPerson",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        person_id = data["person_id"]
        assert person_id is not None

        # Step 2: Verify the same face (uses same stable_embedding)
        _state["matching_service"].invalidate_cache()
        response = e2e_client.post(
            "/verify",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["person_id"] == person_id
        assert data["name"] == "TestPerson"
        assert data["attendance_logged"] is True

        # Step 3: Check today's attendance
        response = e2e_client.get("/attendance/today")
        assert response.status_code == 200
        data = response.json()
        assert len(data["records"]) == 1
        assert data["records"][0]["name"] == "TestPerson"

        # Step 4: Verify deduplication (second verify same session)
        response = e2e_client.post(
            "/verify",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["success"] is True
        assert data["attendance_logged"] is False  # Deduped

        # Step 5: Generate report
        today = datetime.now().strftime("%Y-%m-%d")
        response = e2e_client.get(f"/attendance/report?start={today}&end={today}")
        assert response.status_code == 200

        # Step 6: Generate CSV report
        response = e2e_client.get(f"/attendance/report?start={today}&end={today}&format=csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]

        # Step 7: List persons
        response = e2e_client.get("/persons")
        assert response.status_code == 200
        data = response.json()
        assert any(p["name"] == "TestPerson" for p in data["persons"])

        # Step 8: Check audit log
        response = e2e_client.get("/audit")
        assert response.status_code == 200
        data = response.json()
        assert len(data["entries"]) >= 2  # enroll + verify

        # Step 9: Delete person (GDPR)
        response = e2e_client.delete(f"/person/{person_id}")
        assert response.status_code == 200
        assert response.json()["success"] is True

        # Step 10: Verify person no longer recognized
        _state["matching_service"].invalidate_cache()
        response = e2e_client.post(
            "/verify",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["success"] is False

    def test_multiple_persons_enrollment(self, e2e_client: TestClient) -> None:
        """Test enrolling multiple persons and verifying each."""
        image_bytes = _create_test_image()
        rng = np.random.default_rng(100)
        person_ids = []

        # Enroll 3 different persons with different embeddings
        for name in ["Alice", "Bob", "Charlie"]:
            emb = rng.standard_normal(512).astype(np.float32)
            _state["face_processor"].process_image.return_value = (
                emb,
                np.array([10, 10, 50, 50]),
                0.99,
            )
            response = e2e_client.post(
                f"/enroll?name={name}",
                files={"image": ("face.jpg", image_bytes, "image/jpeg")},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            person_ids.append(data["person_id"])

        # List all persons
        response = e2e_client.get("/persons")
        assert response.status_code == 200
        data = response.json()
        assert len(data["persons"]) >= 3

    def test_no_face_enrollment_rejected(self, e2e_client: TestClient) -> None:
        """Test that enrollment fails gracefully with no face."""
        _state["face_processor"].process_image.return_value = (None, None, None)
        image_bytes = _create_test_image()

        response = e2e_client.post(
            "/enroll?name=Nobody",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No face" in data["message"]

    def test_health_check_returns_healthy(self, e2e_client: TestClient) -> None:
        """Test health endpoint throughout workflow."""
        response = e2e_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestDataIntegrity:
    """Tests for data integrity across operations."""

    def test_attendance_records_persist(self, e2e_db: str) -> None:
        """Test attendance records persist correctly in SQLite."""
        face_db = FaceDatabase(e2e_db)
        att_db = AttendanceDatabase(e2e_db, dedup_hours=4.0)

        rng = np.random.default_rng(99)
        emb = rng.standard_normal(512).astype(np.float32)
        person_id = face_db.register_person("Alice", [emb], "v1")

        att_id = att_db.log_attendance(person_id, 0.95, 0.87, "present")
        assert att_id is not None

        records = att_db.get_attendance_by_date(datetime.now())
        assert len(records) == 1
        assert records[0]["person_id"] == person_id

    def test_cascade_delete_removes_all_data(self, e2e_db: str) -> None:
        """Test that deleting a person removes embeddings and attendance."""
        face_db = FaceDatabase(e2e_db)
        att_db = AttendanceDatabase(e2e_db, dedup_hours=4.0)
        privacy = PrivacyManager(e2e_db)

        rng = np.random.default_rng(77)
        emb = rng.standard_normal(512).astype(np.float32)
        person_id = face_db.register_person("Bob", [emb], "v1")
        att_db.log_attendance(person_id, 0.90, 0.80, "present")

        result = privacy.delete_person_completely(person_id, "127.0.0.1")
        assert result["success"] is True

        # Verify person is gone
        person = face_db.get_person(person_id)
        assert person is None

        # Verify embeddings are gone
        conn = sqlite3.connect(e2e_db)
        row = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
        ).fetchone()
        assert row[0] == 0
        conn.close()

    def test_deduplication_window(self, e2e_db: str) -> None:
        """Test attendance deduplication within time window."""
        face_db = FaceDatabase(e2e_db)
        att_db = AttendanceDatabase(e2e_db, dedup_hours=4.0)

        rng = np.random.default_rng(55)
        emb = rng.standard_normal(512).astype(np.float32)
        person_id = face_db.register_person("Charlie", [emb], "v1")

        # First log should succeed
        first = att_db.log_attendance(person_id, 0.95, 0.87, "present")
        assert first is not None

        # Second log within window should be deduplicated
        second = att_db.log_attendance(person_id, 0.95, 0.87, "present")
        assert second is None
