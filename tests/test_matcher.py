"""Tests for face similarity matching."""

from pathlib import Path

import numpy as np
import pytest

from src.database import init_database
from src.database.face_db import FaceDatabase
from src.matching import MatchingService
from src.matching.matcher import FaceMatcher


@pytest.fixture
def matcher() -> FaceMatcher:
    """Provide a FaceMatcher with default threshold."""
    return FaceMatcher(similarity_threshold=0.7)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Provide an initialized temporary database."""
    path = str(tmp_path / "test.db")
    init_database(path)
    return path


@pytest.fixture
def face_db(db_path: str) -> FaceDatabase:
    """Provide a FaceDatabase instance."""
    return FaceDatabase(db_path)


def _normalized_embedding(seed: int) -> np.ndarray:
    """Generate a normalized random embedding."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


class TestFaceMatcher:
    """Tests for FaceMatcher class."""

    def test_cosine_similarity_identical(self) -> None:
        """Test cosine similarity of identical embeddings is 1.0."""
        emb = _normalized_embedding(0)
        score = FaceMatcher.cosine_similarity(emb, emb)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal(self) -> None:
        """Test cosine similarity of orthogonal vectors is ~0."""
        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0

        score = FaceMatcher.cosine_similarity(emb1, emb2)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_zero_vector(self) -> None:
        """Test cosine similarity with zero vector returns 0."""
        emb1 = np.zeros(512, dtype=np.float32)
        emb2 = _normalized_embedding(0)
        score = FaceMatcher.cosine_similarity(emb1, emb2)
        assert score == 0.0

    def test_find_match_returns_best(self, matcher: FaceMatcher) -> None:
        """Test find_match returns the best match above threshold."""
        query = _normalized_embedding(0)
        db_embeddings = [
            (1, _normalized_embedding(100)),  # Different
            (2, query * 0.99 + _normalized_embedding(1) * 0.01),  # Very similar
        ]

        result = matcher.find_match(query, db_embeddings)
        assert result is not None
        person_id, score = result
        assert person_id == 2
        assert score > 0.7

    def test_find_match_returns_none_below_threshold(self, matcher: FaceMatcher) -> None:
        """Test find_match returns None when no match above threshold."""
        query = _normalized_embedding(0)
        db_embeddings = [
            (1, _normalized_embedding(100)),
            (2, _normalized_embedding(200)),
        ]

        result = matcher.find_match(query, db_embeddings)
        assert result is None

    def test_find_match_empty_database(self, matcher: FaceMatcher) -> None:
        """Test find_match returns None for empty database."""
        query = _normalized_embedding(0)
        result = matcher.find_match(query, [])
        assert result is None

    def test_find_matches_top_k(self, matcher: FaceMatcher) -> None:
        """Test find_matches returns sorted top-k results."""
        query = _normalized_embedding(0)
        similar1 = query * 0.95 + _normalized_embedding(1) * 0.05
        similar2 = query * 0.90 + _normalized_embedding(2) * 0.10

        db_embeddings = [
            (1, similar1),
            (2, similar2),
            (3, _normalized_embedding(300)),  # Not similar
        ]

        results = matcher.find_matches(query, db_embeddings, top_k=2)
        assert len(results) <= 2
        if len(results) == 2:
            assert results[0][1] >= results[1][1]  # Sorted by score

    def test_verify_match_true(self, matcher: FaceMatcher) -> None:
        """Test verify_match returns True for similar embeddings."""
        emb = _normalized_embedding(0)
        is_match, score = matcher.verify_match(emb, emb)
        assert is_match is True
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_verify_match_false(self, matcher: FaceMatcher) -> None:
        """Test verify_match returns False for different embeddings."""
        emb1 = _normalized_embedding(0)
        emb2 = _normalized_embedding(100)
        is_match, score = matcher.verify_match(emb1, emb2)
        assert is_match is False

    def test_threshold_sensitivity(self) -> None:
        """Test matching behavior at different thresholds."""
        query = _normalized_embedding(0)
        similar = query * 0.85 + _normalized_embedding(1) * 0.15
        db_embeddings = [(1, similar)]

        # Low threshold should match
        low_matcher = FaceMatcher(similarity_threshold=0.5)
        assert low_matcher.find_match(query, db_embeddings) is not None

        # High threshold may not match
        high_matcher = FaceMatcher(similarity_threshold=0.99)
        assert high_matcher.find_match(query, db_embeddings) is None


class TestMatchingService:
    """Tests for MatchingService class."""

    def test_identify_found(self, face_db: FaceDatabase) -> None:
        """Test identify returns person when match found."""
        emb = _normalized_embedding(42)
        face_db.register_person("TestPerson", [emb], "v1")

        service = MatchingService(face_db, threshold=0.7)
        result = service.identify(emb)

        assert result is not None
        person, confidence = result
        assert person["name"] == "TestPerson"
        assert confidence == pytest.approx(1.0, abs=1e-5)

    def test_identify_not_found(self, face_db: FaceDatabase) -> None:
        """Test identify returns None when no match."""
        emb = _normalized_embedding(42)
        face_db.register_person("TestPerson", [emb], "v1")

        service = MatchingService(face_db, threshold=0.7)
        query = _normalized_embedding(999)
        result = service.identify(query)

        assert result is None

    def test_cache_works(self, face_db: FaceDatabase) -> None:
        """Test that caching avoids redundant DB queries."""
        emb = _normalized_embedding(42)
        face_db.register_person("TestPerson", [emb], "v1")

        service = MatchingService(face_db, threshold=0.7)

        # First call loads cache
        service.identify(emb)
        assert service._cache_valid is True

        # Second call uses cache
        service.identify(emb)
        assert service._cache_valid is True

    def test_invalidate_cache(self, face_db: FaceDatabase) -> None:
        """Test cache invalidation."""
        emb = _normalized_embedding(42)
        face_db.register_person("TestPerson", [emb], "v1")

        service = MatchingService(face_db, threshold=0.7)
        service.identify(emb)
        assert service._cache_valid is True

        service.invalidate_cache()
        assert service._cache_valid is False
        assert service._embedding_cache is None
