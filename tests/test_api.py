"""Tests for FastAPI REST API endpoints."""

import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import _state, app
from src.database import init_database
from src.database.attendance_db import AttendanceDatabase
from src.database.face_db import FaceDatabase, embedding_to_bytes
from src.detection.liveness import LivenessDetector
from src.matching import MatchingService
from src.reporting.report_generator import ReportGenerator
from src.utils.privacy import PrivacyManager


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Provide an initialized temporary database."""
    path = str(tmp_path / "test.db")
    init_database(path)
    return path


@pytest.fixture
def populated_db(db_path: str) -> str:
    """Database with test persons."""
    conn = sqlite3.connect(db_path)
    rng = np.random.default_rng(0)
    conn.execute("INSERT INTO persons (id, name) VALUES (1, 'Alice')")
    emb = rng.standard_normal(512).astype(np.float32)
    conn.execute(
        "INSERT INTO embeddings (person_id, embedding, model_version) VALUES (?, ?, ?)",
        (1, embedding_to_bytes(emb), "v1"),
    )
    today = datetime.now().strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO attendance (person_id, confidence, status, timestamp) VALUES (?, ?, ?, ?)",
        (1, 0.95, "present", f"{today} 08:50:00"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_face_processor():
    """Mock face processor that returns a fake embedding."""
    processor = MagicMock()
    processor.process_image.return_value = (
        np.random.randn(512).astype(np.float32),
        np.array([10, 10, 50, 50]),
        0.99,
    )
    processor.embedder.get_model_version.return_value = "v1"
    return processor


@pytest.fixture
def client(populated_db: str, mock_face_processor: MagicMock) -> TestClient:
    """Provide a test client with all components initialized."""
    face_db = FaceDatabase(populated_db)
    attendance_db = AttendanceDatabase(populated_db, dedup_hours=4.0)

    _state["face_processor"] = mock_face_processor
    _state["face_db"] = face_db
    _state["attendance_db"] = attendance_db
    _state["matching_service"] = MatchingService(face_db, threshold=0.7)
    _state["liveness_detector"] = LivenessDetector(texture_threshold=0.3)
    _state["privacy_manager"] = PrivacyManager(populated_db)
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


def _create_test_image() -> bytes:
    """Create a simple test image as bytes."""
    import io

    img = Image.fromarray(np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestEnrollEndpoint:
    """Tests for POST /enroll."""

    def test_enroll_success(self, client: TestClient) -> None:
        """Test successful enrollment."""
        image_bytes = _create_test_image()
        response = client.post(
            "/enroll?name=Bob",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["person_id"] is not None

    def test_enroll_no_face(self, client: TestClient, mock_face_processor: MagicMock) -> None:
        """Test enrollment with no face detected."""
        mock_face_processor.process_image.return_value = (None, None, None)

        image_bytes = _create_test_image()
        response = client.post(
            "/enroll?name=Nobody",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No face" in data["message"]


class TestVerifyEndpoint:
    """Tests for POST /verify."""

    def test_verify_known_face(self, client: TestClient) -> None:
        """Test verification of a known face."""
        # Use Alice's existing embedding for matching
        emb = _state["face_db"].get_all_embeddings()[0][1]
        _state["face_processor"].process_image.return_value = (
            emb,
            np.array([10, 10, 50, 50]),
            0.99,
        )
        _state["matching_service"].invalidate_cache()

        image_bytes = _create_test_image()
        response = client.post(
            "/verify",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["name"] == "Alice"

    def test_verify_no_face(self, client: TestClient, mock_face_processor: MagicMock) -> None:
        """Test verification with no face."""
        mock_face_processor.process_image.return_value = (None, None, None)

        image_bytes = _create_test_image()
        response = client.post(
            "/verify",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["success"] is False


class TestAttendanceEndpoints:
    """Tests for attendance endpoints."""

    def test_get_today_attendance(self, client: TestClient) -> None:
        """Test getting today's attendance."""
        response = client.get("/attendance/today")
        assert response.status_code == 200
        data = response.json()
        assert "date" in data
        assert "records" in data
        assert "summary" in data

    def test_get_attendance_report_json(self, client: TestClient) -> None:
        """Test attendance report in JSON format."""
        today = datetime.now().strftime("%Y-%m-%d")
        response = client.get(f"/attendance/report?start={today}&end={today}")
        assert response.status_code == 200

    def test_get_attendance_report_csv(self, client: TestClient) -> None:
        """Test attendance report in CSV format."""
        today = datetime.now().strftime("%Y-%m-%d")
        response = client.get(f"/attendance/report?start={today}&end={today}&format=csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]

    def test_get_attendance_report_markdown(self, client: TestClient) -> None:
        """Test attendance report in Markdown format."""
        today = datetime.now().strftime("%Y-%m-%d")
        response = client.get(f"/attendance/report?start={today}&end={today}&format=markdown")
        assert response.status_code == 200


class TestDeleteEndpoint:
    """Tests for DELETE /person/{id}."""

    def test_delete_person(self, client: TestClient) -> None:
        """Test deleting a person."""
        response = client.delete("/person/1")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_delete_nonexistent(self, client: TestClient) -> None:
        """Test deleting non-existent person."""
        response = client.delete("/person/999")
        assert response.status_code == 404


class TestAuditEndpoint:
    """Tests for GET /audit."""

    def test_get_audit_log(self, client: TestClient) -> None:
        """Test getting audit log."""
        response = client.get("/audit")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data


class TestPersonsEndpoint:
    """Tests for GET /persons."""

    def test_list_persons(self, client: TestClient) -> None:
        """Test listing persons."""
        response = client.get("/persons")
        assert response.status_code == 200
        data = response.json()
        assert "persons" in data
        assert len(data["persons"]) >= 1
