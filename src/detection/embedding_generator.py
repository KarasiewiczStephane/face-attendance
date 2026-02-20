"""FaceNet embedding generation using InceptionResnetV1.

Generates 512-dimensional face embeddings from aligned face tensors
using a pretrained FaceNet model.
"""

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """Generate 512-d face embeddings using FaceNet.

    Args:
        pretrained: Pretrained model name ('vggface2' or 'casia-webface').
        device: Computation device ('cpu' or 'cuda').
    """

    MODEL_VERSION = "facenet-vggface2-v1"
    EMBEDDING_DIM = 512

    def __init__(self, pretrained: str = "vggface2", device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,
            device=self.device,
        ).eval()
        logger.info("EmbeddingGenerator initialized with %s on %s", pretrained, device)

    @torch.no_grad()
    def generate_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Generate embedding for a single aligned face tensor.

        Args:
            face_tensor: Preprocessed face tensor of shape (3, 160, 160)
                or (1, 3, 160, 160).

        Returns:
            512-d embedding as numpy array.
        """
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)

        face_tensor = face_tensor.to(self.device)
        embedding = self.model(face_tensor)
        return embedding.cpu().numpy().flatten()

    @torch.no_grad()
    def generate_embeddings_batch(self, face_tensors: torch.Tensor) -> np.ndarray:
        """Generate embeddings for a batch of faces.

        Args:
            face_tensors: Batch of faces of shape (N, 3, 160, 160).

        Returns:
            (N, 512) numpy array of embeddings.
        """
        face_tensors = face_tensors.to(self.device)
        embeddings = self.model(face_tensors)
        return embeddings.cpu().numpy()

    def get_model_version(self) -> str:
        """Return model version string for embedding versioning.

        Returns:
            Model version identifier.
        """
        return self.MODEL_VERSION
