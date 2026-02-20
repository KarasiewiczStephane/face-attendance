"""Tests for MTCNN face detection."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.detection.face_detector import FaceDetector


@pytest.fixture
def mock_mtcnn():
    """Patch MTCNN to avoid loading actual model weights."""
    with patch("src.detection.face_detector.MTCNN") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def detector(mock_mtcnn: MagicMock) -> FaceDetector:
    """Create a FaceDetector with mocked MTCNN."""
    return FaceDetector(image_size=160, device="cpu")


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample RGB PIL image."""
    return Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))


class TestFaceDetector:
    """Tests for FaceDetector class."""

    def test_init_default_thresholds(self, mock_mtcnn: MagicMock) -> None:
        """Test detector initializes with default thresholds."""
        detector = FaceDetector()
        assert detector.image_size == 160

    def test_detect_faces_returns_none_when_no_faces(
        self,
        detector: FaceDetector,
        mock_mtcnn: MagicMock,
        sample_image: Image.Image,
    ) -> None:
        """Test detection returns None tuple when no faces found."""
        mock_mtcnn.return_value = (None, None)
        mock_mtcnn.detect.return_value = (None, None)

        faces, boxes, probs = detector.detect_faces(sample_image)

        assert faces is None
        assert boxes is None
        assert probs is None

    def test_detect_faces_with_numpy_input(
        self,
        detector: FaceDetector,
        mock_mtcnn: MagicMock,
    ) -> None:
        """Test detection accepts numpy array input."""
        img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mock_mtcnn.return_value = (None, None)
        mock_mtcnn.detect.return_value = (None, None)

        faces, boxes, probs = detector.detect_faces(img_array)
        assert faces is None

    def test_detect_faces_returns_tensors(
        self,
        detector: FaceDetector,
        mock_mtcnn: MagicMock,
        sample_image: Image.Image,
    ) -> None:
        """Test detection returns proper tensor shapes."""
        face_tensor = torch.randn(2, 3, 160, 160)
        prob_array = np.array([0.99, 0.95])
        box_array = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])

        mock_mtcnn.return_value = (face_tensor, prob_array)
        mock_mtcnn.detect.return_value = (box_array, None)

        faces, boxes, probs = detector.detect_faces(sample_image)

        assert faces is not None
        assert faces.shape == (2, 3, 160, 160)
        assert boxes is not None
        assert len(boxes) == 2

    def test_detect_single_face_returns_best(
        self,
        detector: FaceDetector,
        mock_mtcnn: MagicMock,
        sample_image: Image.Image,
    ) -> None:
        """Test single face detection returns highest confidence."""
        face_tensor = torch.randn(2, 3, 160, 160)
        prob_array = np.array([0.80, 0.99])
        box_array = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])

        mock_mtcnn.return_value = (face_tensor, prob_array)
        mock_mtcnn.detect.return_value = (box_array, None)

        face, box, prob = detector.detect_single_face(sample_image)

        assert face is not None
        assert face.shape == (3, 160, 160)
        assert prob == pytest.approx(0.99)

    def test_detect_single_face_none(
        self,
        detector: FaceDetector,
        mock_mtcnn: MagicMock,
        sample_image: Image.Image,
    ) -> None:
        """Test single face detection returns None when no face."""
        mock_mtcnn.return_value = (None, None)
        mock_mtcnn.detect.return_value = (None, None)

        face, box, prob = detector.detect_single_face(sample_image)

        assert face is None
        assert box is None
        assert prob is None

    def test_detect_single_face_when_only_one(
        self,
        detector: FaceDetector,
        mock_mtcnn: MagicMock,
        sample_image: Image.Image,
    ) -> None:
        """Test single face detection when exactly one face found."""
        face_tensor = torch.randn(3, 160, 160)  # 3D tensor = single face
        prob_array = np.array([0.95])
        box_array = np.array([[10, 10, 50, 50]])

        mock_mtcnn.return_value = (face_tensor, prob_array)
        mock_mtcnn.detect.return_value = (box_array, None)

        face, box, prob = detector.detect_single_face(sample_image)

        assert face is not None
        assert face.shape == (3, 160, 160)
