"""Tests for face database CRUD operations."""

from pathlib import Path

import numpy as np
import pytest

from src.database import init_database
from src.database.face_db import (
    FaceDatabase,
    bytes_to_embedding,
    embedding_to_bytes,
)


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


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Provide a random 512-d embedding."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(512).astype(np.float32)
    return emb


class TestSerializtion:
    """Tests for embedding serialization."""

    def test_roundtrip(self, sample_embedding: np.ndarray) -> None:
        """Test embedding survives serialization roundtrip."""
        data = embedding_to_bytes(sample_embedding)
        restored = bytes_to_embedding(data)
        np.testing.assert_array_almost_equal(sample_embedding, restored)

    def test_bytes_length(self, sample_embedding: np.ndarray) -> None:
        """Test serialized bytes have correct length for float32."""
        data = embedding_to_bytes(sample_embedding)
        assert len(data) == 512 * 4  # float32 = 4 bytes


class TestFaceDatabase:
    """Tests for FaceDatabase CRUD operations."""

    def test_register_person(
        self,
        face_db: FaceDatabase,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test registering a person creates person and embedding."""
        person_id = face_db.register_person("Alice", [sample_embedding], "v1")

        assert person_id > 0
        person = face_db.get_person(person_id)
        assert person is not None
        assert person["name"] == "Alice"

    def test_register_person_multiple_embeddings(
        self,
        face_db: FaceDatabase,
    ) -> None:
        """Test registering a person with multiple embeddings."""
        rng = np.random.default_rng(0)
        embeddings = [rng.standard_normal(512).astype(np.float32) for _ in range(3)]
        person_id = face_db.register_person("Bob", embeddings, "v1")

        stored = face_db.get_person_embeddings(person_id)
        assert len(stored) == 3

    def test_add_embedding(
        self,
        face_db: FaceDatabase,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test adding embedding to existing person."""
        person_id = face_db.register_person("Carol", [sample_embedding], "v1")

        new_emb = np.random.randn(512).astype(np.float32)
        emb_id = face_db.add_embedding(person_id, new_emb, "v1")

        assert emb_id > 0
        embeddings = face_db.get_person_embeddings(person_id)
        assert len(embeddings) == 2

    def test_get_person_returns_none_for_missing(
        self,
        face_db: FaceDatabase,
    ) -> None:
        """Test get_person returns None for non-existent ID."""
        assert face_db.get_person(999) is None

    def test_get_all_embeddings(
        self,
        face_db: FaceDatabase,
    ) -> None:
        """Test get_all_embeddings returns embeddings for active persons."""
        rng = np.random.default_rng(1)
        emb1 = rng.standard_normal(512).astype(np.float32)
        emb2 = rng.standard_normal(512).astype(np.float32)

        face_db.register_person("Alice", [emb1], "v1")
        face_db.register_person("Bob", [emb2], "v1")

        all_embeddings = face_db.get_all_embeddings()
        assert len(all_embeddings) == 2
        assert all_embeddings[0][0] != all_embeddings[1][0]  # Different person IDs

    def test_delete_person_cascade(
        self,
        face_db: FaceDatabase,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test deleting a person removes embeddings via cascade."""
        person_id = face_db.register_person("DeleteMe", [sample_embedding], "v1")

        assert face_db.delete_person(person_id) is True
        assert face_db.get_person(person_id) is None
        assert face_db.get_person_embeddings(person_id) == []

    def test_delete_nonexistent_person(
        self,
        face_db: FaceDatabase,
    ) -> None:
        """Test deleting non-existent person returns False."""
        assert face_db.delete_person(999) is False

    def test_list_persons(
        self,
        face_db: FaceDatabase,
    ) -> None:
        """Test listing active persons."""
        rng = np.random.default_rng(2)
        face_db.register_person("Alice", [rng.standard_normal(512).astype(np.float32)], "v1")
        face_db.register_person("Bob", [rng.standard_normal(512).astype(np.float32)], "v1")

        persons = face_db.list_persons()
        assert len(persons) == 2
        names = {p["name"] for p in persons}
        assert names == {"Alice", "Bob"}

    def test_update_person_name(
        self,
        face_db: FaceDatabase,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test updating a person's name."""
        person_id = face_db.register_person("OldName", [sample_embedding], "v1")

        assert face_db.update_person_name(person_id, "NewName") is True
        person = face_db.get_person(person_id)
        assert person["name"] == "NewName"

    def test_update_nonexistent_person(
        self,
        face_db: FaceDatabase,
    ) -> None:
        """Test updating non-existent person returns False."""
        assert face_db.update_person_name(999, "Nobody") is False

    def test_embedding_data_preserved(
        self,
        face_db: FaceDatabase,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test that stored embeddings match the original data."""
        person_id = face_db.register_person("TestEmb", [sample_embedding], "v1")

        stored = face_db.get_person_embeddings(person_id)
        assert len(stored) == 1
        np.testing.assert_array_almost_equal(stored[0], sample_embedding)

    def test_auto_creates_db(self, tmp_path: Path) -> None:
        """Test FaceDatabase auto-creates database if missing."""
        db_path = str(tmp_path / "auto" / "face.db")
        db = FaceDatabase(db_path)

        assert Path(db_path).exists()
        rng = np.random.default_rng(3)
        person_id = db.register_person("Auto", [rng.standard_normal(512).astype(np.float32)], "v1")
        assert person_id > 0
