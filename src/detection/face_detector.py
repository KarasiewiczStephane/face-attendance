"""MTCNN-based face detection with alignment.

Uses facenet-pytorch's MTCNN implementation for multi-task cascaded
face detection, returning aligned face tensors ready for embedding.
"""

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceDetector:
    """MTCNN-based face detection with alignment.

    Detects faces in images and returns aligned, preprocessed face
    tensors suitable for embedding generation.

    Args:
        image_size: Output face image size (default: 160).
        min_face_size: Minimum face size in pixels to detect.
        thresholds: MTCNN stage thresholds [P-Net, R-Net, O-Net].
        device: Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        image_size: int = 160,
        min_face_size: int = 20,
        thresholds: list[float] | None = None,
        device: str = "cpu",
    ) -> None:
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]
        self.device = torch.device(device)
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=0,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True,
        )
        self.image_size = image_size
        logger.info("FaceDetector initialized on %s", device)

    def detect_faces(
        self,
        image: np.ndarray | Image.Image,
    ) -> tuple[torch.Tensor | None, np.ndarray | None, np.ndarray | None]:
        """Detect all faces in an image.

        Args:
            image: Input image as numpy array (BGR/RGB) or PIL Image.

        Returns:
            Tuple of (aligned_faces, bounding_boxes, probabilities).
            All None if no faces detected.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        faces, probs = self.mtcnn(image, return_prob=True)
        boxes, _ = self.mtcnn.detect(image)

        if faces is None:
            return None, None, None

        return faces, boxes, probs

    def detect_single_face(
        self,
        image: np.ndarray | Image.Image,
    ) -> tuple[torch.Tensor | None, np.ndarray | None, float | None]:
        """Detect and return the most confident face.

        Args:
            image: Input image as numpy array (BGR/RGB) or PIL Image.

        Returns:
            Tuple of (aligned_face, bounding_box, probability).
            All None if no face detected.
        """
        faces, boxes, probs = self.detect_faces(image)

        if faces is None:
            return None, None, None

        if faces.dim() == 3:
            return faces, boxes[0], float(probs[0]) if probs is not None else None

        best_idx = int(probs.argmax()) if probs is not None else 0
        return faces[best_idx], boxes[best_idx], float(probs[best_idx])
