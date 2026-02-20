"""Performance benchmark tests for core operations."""

import time
from statistics import mean

import numpy as np
import pytest

from src.database import init_database
from src.database.face_db import FaceDatabase
from src.matching.matcher import FaceMatcher


@pytest.fixture
def face_db(tmp_path) -> FaceDatabase:
    """Provide an initialized face database."""
    path = str(tmp_path / "perf.db")
    init_database(path)
    return FaceDatabase(path)


@pytest.fixture
def db_embeddings(face_db: FaceDatabase) -> list[tuple[int, np.ndarray]]:
    """Create 100 enrolled embeddings and return them."""
    rng = np.random.default_rng(0)
    for i in range(100):
        emb = rng.standard_normal(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        face_db.register_person(f"Person_{i}", [emb], "v1")

    return face_db.get_all_embeddings()


class TestMatchingPerformance:
    """Performance tests for face matching."""

    def test_cosine_similarity_computation(self) -> None:
        """Cosine similarity between two 512-d vectors should be fast."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal(512).astype(np.float32)
        b = rng.standard_normal(512).astype(np.float32)

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            FaceMatcher.cosine_similarity(a, b)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_ms = mean(latencies)
        assert avg_ms < 1.0, f"Cosine similarity too slow: {avg_ms:.3f}ms"

    def test_matching_against_100_embeddings(
        self, db_embeddings: list[tuple[int, np.ndarray]]
    ) -> None:
        """Matching against 100 embeddings should complete in <10ms."""
        matcher = FaceMatcher(similarity_threshold=0.7)
        rng = np.random.default_rng(99)
        query = rng.standard_normal(512).astype(np.float32)

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            matcher.find_match(query, db_embeddings)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_ms = mean(latencies)
        assert avg_ms < 10.0, f"Matching too slow: {avg_ms:.1f}ms"

    def test_batch_matching(self, db_embeddings: list[tuple[int, np.ndarray]]) -> None:
        """Finding top-k matches should complete quickly."""
        matcher = FaceMatcher(similarity_threshold=0.7)
        rng = np.random.default_rng(88)
        query = rng.standard_normal(512).astype(np.float32)

        start = time.perf_counter()
        matches = matcher.find_matches(query, db_embeddings, top_k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10.0, f"Batch matching too slow: {elapsed_ms:.1f}ms"
        assert len(matches) <= 5


class TestDatabasePerformance:
    """Performance tests for database operations."""

    def test_embedding_retrieval_100(self, face_db: FaceDatabase) -> None:
        """Retrieving 100 embeddings should be fast."""
        rng = np.random.default_rng(0)
        for i in range(100):
            emb = rng.standard_normal(512).astype(np.float32)
            face_db.register_person(f"Person_{i}", [emb], "v1")

        start = time.perf_counter()
        embeddings = face_db.get_all_embeddings()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(embeddings) == 100
        assert elapsed_ms < 100.0, f"Retrieval too slow: {elapsed_ms:.1f}ms"

    def test_person_registration(self, face_db: FaceDatabase) -> None:
        """Registering a person should complete in <50ms."""
        rng = np.random.default_rng(0)

        latencies = []
        for i in range(20):
            emb = rng.standard_normal(512).astype(np.float32)
            start = time.perf_counter()
            face_db.register_person(f"Person_{i}", [emb], "v1")
            latencies.append((time.perf_counter() - start) * 1000)

        avg_ms = mean(latencies)
        assert avg_ms < 50.0, f"Registration too slow: {avg_ms:.1f}ms"


class TestEmbeddingQuality:
    """Tests for embedding quality and matching correctness."""

    def test_same_embedding_gives_perfect_match(self) -> None:
        """A vector matched against itself should give similarity ~1.0."""
        rng = np.random.default_rng(0)
        emb = rng.standard_normal(512).astype(np.float32)
        sim = FaceMatcher.cosine_similarity(emb, emb)
        assert sim > 0.999

    def test_random_embeddings_low_similarity(self) -> None:
        """Random embeddings should have low similarity."""
        rng = np.random.default_rng(0)
        sims = []
        for _ in range(100):
            a = rng.standard_normal(512).astype(np.float32)
            b = rng.standard_normal(512).astype(np.float32)
            sims.append(FaceMatcher.cosine_similarity(a, b))

        avg_sim = mean(sims)
        assert abs(avg_sim) < 0.15, f"Random similarity too high: {avg_sim:.3f}"

    def test_similar_embeddings_high_similarity(self) -> None:
        """Slightly perturbed embeddings should retain high similarity."""
        rng = np.random.default_rng(0)
        base = rng.standard_normal(512).astype(np.float32)
        noise = rng.standard_normal(512).astype(np.float32) * 0.05
        perturbed = base + noise

        sim = FaceMatcher.cosine_similarity(base, perturbed)
        assert sim > 0.9, f"Similar embeddings should match: {sim:.3f}"
