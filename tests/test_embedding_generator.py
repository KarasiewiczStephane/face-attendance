"""Tests for FaceNet embedding generation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.detection.embedding_generator import EmbeddingGenerator


@pytest.fixture
def mock_inception():
    """Patch InceptionResnetV1 to avoid loading actual model weights."""
    with patch("src.detection.embedding_generator.InceptionResnetV1") as mock_cls:
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_cls.return_value = mock_model
        yield mock_model


@pytest.fixture
def embedder(mock_inception: MagicMock) -> EmbeddingGenerator:
    """Create an EmbeddingGenerator with mocked model."""
    return EmbeddingGenerator(pretrained="vggface2", device="cpu")


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    def test_generate_embedding_shape(
        self,
        embedder: EmbeddingGenerator,
        mock_inception: MagicMock,
    ) -> None:
        """Test embedding output shape is (512,)."""
        mock_inception.return_value = torch.randn(1, 512)

        face_tensor = torch.randn(3, 160, 160)
        embedding = embedder.generate_embedding(face_tensor)

        assert embedding.shape == (512,)
        assert isinstance(embedding, np.ndarray)

    def test_generate_embedding_4d_input(
        self,
        embedder: EmbeddingGenerator,
        mock_inception: MagicMock,
    ) -> None:
        """Test embedding accepts 4D tensor input."""
        mock_inception.return_value = torch.randn(1, 512)

        face_tensor = torch.randn(1, 3, 160, 160)
        embedding = embedder.generate_embedding(face_tensor)

        assert embedding.shape == (512,)

    def test_generate_embeddings_batch(
        self,
        embedder: EmbeddingGenerator,
        mock_inception: MagicMock,
    ) -> None:
        """Test batch embedding generation shape."""
        batch_size = 4
        mock_inception.return_value = torch.randn(batch_size, 512)

        face_tensors = torch.randn(batch_size, 3, 160, 160)
        embeddings = embedder.generate_embeddings_batch(face_tensors)

        assert embeddings.shape == (batch_size, 512)
        assert isinstance(embeddings, np.ndarray)

    def test_get_model_version(self, embedder: EmbeddingGenerator) -> None:
        """Test model version string is returned."""
        version = embedder.get_model_version()
        assert version == "facenet-vggface2-v1"
        assert isinstance(version, str)

    def test_embedding_dim_constant(self) -> None:
        """Test embedding dimension constant is 512."""
        assert EmbeddingGenerator.EMBEDDING_DIM == 512

    def test_embedding_deterministic(
        self,
        embedder: EmbeddingGenerator,
        mock_inception: MagicMock,
    ) -> None:
        """Test same input produces same output (no randomness)."""
        fixed_output = torch.randn(1, 512)
        mock_inception.return_value = fixed_output

        face_tensor = torch.randn(3, 160, 160)
        emb1 = embedder.generate_embedding(face_tensor)
        emb2 = embedder.generate_embedding(face_tensor)

        np.testing.assert_array_equal(emb1, emb2)


class TestFaceProcessor:
    """Tests for the combined FaceProcessor pipeline."""

    @patch("src.detection.EmbeddingGenerator")
    @patch("src.detection.FaceDetector")
    def test_process_image_returns_embedding(
        self,
        mock_detector_cls: MagicMock,
        mock_embedder_cls: MagicMock,
    ) -> None:
        """Test process_image returns embedding when face found."""
        from src.detection import FaceProcessor

        mock_detector = MagicMock()
        mock_detector.detect_single_face.return_value = (
            torch.randn(3, 160, 160),
            np.array([10, 10, 50, 50]),
            0.99,
        )
        mock_detector_cls.return_value = mock_detector

        mock_embedder = MagicMock()
        mock_embedder.generate_embedding.return_value = np.random.randn(512)
        mock_embedder_cls.return_value = mock_embedder

        processor = FaceProcessor({"face_detection": {}, "embedding": {}})
        # Manually assign mocks since constructor creates new instances
        processor.detector = mock_detector
        processor.embedder = mock_embedder

        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        embedding, bbox, prob = processor.process_image(image)

        assert embedding is not None
        assert embedding.shape == (512,)

    @patch("src.detection.EmbeddingGenerator")
    @patch("src.detection.FaceDetector")
    def test_process_image_no_face(
        self,
        mock_detector_cls: MagicMock,
        mock_embedder_cls: MagicMock,
    ) -> None:
        """Test process_image returns None when no face detected."""
        from src.detection import FaceProcessor

        mock_detector = MagicMock()
        mock_detector.detect_single_face.return_value = (None, None, None)
        mock_detector_cls.return_value = mock_detector
        mock_embedder_cls.return_value = MagicMock()

        processor = FaceProcessor({"face_detection": {}, "embedding": {}})
        processor.detector = mock_detector

        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        embedding, bbox, prob = processor.process_image(image)

        assert embedding is None
        assert bbox is None
        assert prob is None
