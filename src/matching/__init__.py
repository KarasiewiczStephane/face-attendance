"""Face matching and identification services.

Provides high-level matching service combining database lookups
with cosine similarity scoring.
"""

import numpy as np

from ..database.face_db import FaceDatabase
from ..utils.logger import setup_logger
from .matcher import FaceMatcher

logger = setup_logger(__name__)

__all__ = ["FaceMatcher", "MatchingService"]


class MatchingService:
    """High-level matching service combining database and matcher.

    Caches database embeddings for efficient repeated matching
    and provides person identification from embeddings.

    Args:
        face_db: FaceDatabase instance for embedding retrieval.
        threshold: Cosine similarity threshold for matching.
    """

    def __init__(self, face_db: FaceDatabase, threshold: float = 0.7) -> None:
        self.face_db = face_db
        self.matcher = FaceMatcher(similarity_threshold=threshold)
        self._embedding_cache: list[tuple[int, np.ndarray]] | None = None
        self._cache_valid = False

    def invalidate_cache(self) -> None:
        """Invalidate embedding cache (call after DB changes)."""
        self._cache_valid = False
        self._embedding_cache = None

    def _load_embeddings(self) -> list[tuple[int, np.ndarray]]:
        """Load all embeddings from database with caching.

        Returns:
            List of (person_id, embedding) tuples.
        """
        if not self._cache_valid:
            self._embedding_cache = self.face_db.get_all_embeddings()
            self._cache_valid = True
        return self._embedding_cache

    def identify(
        self,
        query_embedding: np.ndarray,
    ) -> tuple[dict, float] | None:
        """Identify a person from their face embedding.

        Args:
            query_embedding: 512-d face embedding to identify.

        Returns:
            (person_dict, confidence) if match found, None otherwise.
        """
        embeddings = self._load_embeddings()
        match = self.matcher.find_match(query_embedding, embeddings)

        if match:
            person_id, confidence = match
            person = self.face_db.get_person(person_id)
            if person:
                return (person, confidence)

        return None
