"""Face detection and embedding pipeline.

Provides MTCNN-based face detection, FaceNet embedding generation,
and a combined processor for end-to-end face processing.
"""

import numpy as np
from PIL import Image

from ..utils.logger import setup_logger
from .embedding_generator import EmbeddingGenerator
from .face_detector import FaceDetector

logger = setup_logger(__name__)

__all__ = ["FaceDetector", "EmbeddingGenerator", "FaceProcessor"]


class FaceProcessor:
    """Combined face detection and embedding pipeline.

    Wraps FaceDetector and EmbeddingGenerator for convenient
    end-to-end face processing from image to embedding.

    Args:
        config: Configuration dictionary with face_detection and embedding sections.
    """

    def __init__(self, config: dict) -> None:
        det_cfg = config.get("face_detection", {})
        emb_cfg = config.get("embedding", {})

        self.detector = FaceDetector(
            image_size=emb_cfg.get("image_size", 160),
            min_face_size=det_cfg.get("min_face_size", 20),
            thresholds=det_cfg.get("thresholds"),
            device=det_cfg.get("device", "cpu"),
        )
        self.embedder = EmbeddingGenerator(
            pretrained=emb_cfg.get("model", "vggface2"),
            device=det_cfg.get("device", "cpu"),
        )

    def process_image(
        self,
        image: np.ndarray | Image.Image,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        """Detect face and generate embedding from an image.

        Args:
            image: Input image as numpy array or PIL Image.

        Returns:
            Tuple of (embedding, bounding_box, probability).
            All None if no face detected.
        """
        face, bbox, prob = self.detector.detect_single_face(image)
        if face is None:
            return None, None, None

        embedding = self.embedder.generate_embedding(face)
        return embedding, bbox, prob
