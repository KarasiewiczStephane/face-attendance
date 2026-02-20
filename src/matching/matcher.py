"""Face embedding similarity matching.

Provides cosine similarity matching between face embeddings with
configurable thresholds for face identification and verification.
"""

import numpy as np
from numpy.linalg import norm

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceMatcher:
    """Face embedding similarity matching.

    Computes cosine similarity between embeddings and finds the best
    match above a configurable threshold.

    Args:
        similarity_threshold: Minimum similarity score to consider a match.
    """

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity score in range [-1, 1].
        """
        n1 = norm(emb1)
        n2 = norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    def find_match(
        self,
        query_embedding: np.ndarray,
        database_embeddings: list[tuple[int, np.ndarray]],
    ) -> tuple[int, float] | None:
        """Find the best matching person from database.

        Args:
            query_embedding: 512-d embedding to match.
            database_embeddings: List of (person_id, embedding) tuples.

        Returns:
            (person_id, similarity_score) if match found, None otherwise.
        """
        if not database_embeddings:
            return None

        best_match = None
        best_score = -1.0

        for person_id, db_embedding in database_embeddings:
            score = self.cosine_similarity(query_embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_match = person_id

        if best_score >= self.similarity_threshold:
            return (best_match, best_score)

        return None

    def find_matches(
        self,
        query_embedding: np.ndarray,
        database_embeddings: list[tuple[int, np.ndarray]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find top-k matches above threshold.

        Args:
            query_embedding: 512-d embedding to match.
            database_embeddings: List of (person_id, embedding) tuples.
            top_k: Maximum number of matches to return.

        Returns:
            List of (person_id, score) tuples sorted by score descending.
        """
        scores = []

        for person_id, db_embedding in database_embeddings:
            score = self.cosine_similarity(query_embedding, db_embedding)
            if score >= self.similarity_threshold:
                scores.append((person_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def verify_match(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> tuple[bool, float]:
        """Verify if two embeddings are from the same person.

        Args:
            embedding1: First face embedding.
            embedding2: Second face embedding.

        Returns:
            Tuple of (is_match, similarity_score).
        """
        score = self.cosine_similarity(embedding1, embedding2)
        return (score >= self.similarity_threshold, score)
