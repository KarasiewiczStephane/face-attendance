"""Tests for webcam demo mode (with mocked camera and display)."""

from unittest.mock import patch

import numpy as np

from src.demo.webcam_demo import WebcamDemo


class TestDrawResults:
    """Tests for WebcamDemo.draw_results (no camera needed)."""

    @patch("src.demo.webcam_demo.FaceProcessor")
    @patch("src.demo.webcam_demo.FaceDatabase")
    @patch("src.demo.webcam_demo.AttendanceDatabase")
    @patch("src.demo.webcam_demo.MatchingService")
    @patch("src.demo.webcam_demo.LivenessDetector")
    @patch("src.demo.webcam_demo.get_settings")
    def _create_demo(
        self,
        mock_settings,
        mock_liveness,
        mock_matching,
        mock_att_db,
        mock_face_db,
        mock_processor,
    ) -> WebcamDemo:
        """Create a WebcamDemo with all dependencies mocked."""
        mock_settings.return_value.load_config.return_value = {
            "face_detection": {"device": "cpu"},
            "embedding": {"model": "vggface2"},
            "matching": {"similarity_threshold": 0.7},
            "liveness": {"blink_threshold": 0.25, "texture_threshold": 0.5},
            "attendance": {
                "dedup_window_hours": 4,
                "work_start": "09:00",
                "late_threshold_minutes": 15,
            },
            "database": {"path": ":memory:"},
        }
        return WebcamDemo()

    def test_draw_no_face(self) -> None:
        """Test drawing with no face detected."""
        demo = self._create_demo()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = demo.draw_results(frame, None, None, None, None, False)
        assert result.shape == (480, 640, 3)

    def test_draw_recognized(self) -> None:
        """Test drawing with recognized face."""
        demo = self._create_demo()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200])
        liveness = {"confidence": 0.9, "overall_live": True}

        result = demo.draw_results(frame, bbox, "Alice", 0.95, liveness, True)
        assert result.shape == (480, 640, 3)

    def test_draw_unknown(self) -> None:
        """Test drawing with unknown face."""
        demo = self._create_demo()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200])

        result = demo.draw_results(frame, bbox, None, None, None, False)
        assert result.shape == (480, 640, 3)

    def test_draw_with_liveness_failed(self) -> None:
        """Test drawing with failed liveness check."""
        demo = self._create_demo()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200])
        liveness = {"confidence": 0.2, "overall_live": False}

        result = demo.draw_results(frame, bbox, None, None, liveness, False)
        assert result.shape == (480, 640, 3)
