"""Face database management with CRUD operations.

SQLite-backed storage for persons and their face embeddings, supporting
multiple embeddings per person, versioning, and GDPR-compliant deletion.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a 512-d embedding to bytes.

    Args:
        embedding: Numpy array of shape (512,).

    Returns:
        Raw float32 bytes representation.
    """
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes to a 512-d embedding.

    Args:
        data: Raw float32 bytes.

    Returns:
        Numpy array of shape (512,).
    """
    return np.frombuffer(data, dtype=np.float32).copy()


class FaceDatabase:
    """SQLite-backed face database with CRUD operations.

    Manages persons and their face embeddings, supporting multiple
    embeddings per person and embedding version tracking.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Initialize database if it doesn't exist."""
        from . import init_database

        if not Path(self.db_path).exists():
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            init_database(self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory enabled.

        Returns:
            SQLite connection with Row factory.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def register_person(
        self,
        name: str,
        embeddings: list[np.ndarray],
        model_version: str,
    ) -> int:
        """Register a new person with one or more face embeddings.

        Args:
            name: Person's name.
            embeddings: List of face embedding arrays.
            model_version: Version of the model used to generate embeddings.

        Returns:
            person_id of the newly created person.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
            person_id = cursor.lastrowid

            for emb in embeddings:
                cursor.execute(
                    "INSERT INTO embeddings (person_id, embedding, model_version) VALUES (?, ?, ?)",
                    (person_id, embedding_to_bytes(emb), model_version),
                )

            conn.commit()
            logger.info("Registered person '%s' with id=%d", name, person_id)
            return person_id
        finally:
            conn.close()

    def add_embedding(
        self,
        person_id: int,
        embedding: np.ndarray,
        model_version: str,
    ) -> int:
        """Add a new embedding for an existing person.

        Args:
            person_id: ID of the person.
            embedding: Face embedding array.
            model_version: Version of the model used.

        Returns:
            ID of the new embedding record.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO embeddings (person_id, embedding, model_version) VALUES (?, ?, ?)",
                (person_id, embedding_to_bytes(embedding), model_version),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_person(self, person_id: int) -> dict | None:
        """Get person by ID.

        Args:
            person_id: ID of the person to retrieve.

        Returns:
            Person dict or None if not found/inactive.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT * FROM persons WHERE id = ? AND is_active = 1",
                (person_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_person_embeddings(self, person_id: int) -> list[np.ndarray]:
        """Get all embeddings for a person.

        Args:
            person_id: ID of the person.

        Returns:
            List of embedding arrays.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT embedding FROM embeddings WHERE person_id = ?",
                (person_id,),
            )
            rows = cursor.fetchall()
            return [bytes_to_embedding(row["embedding"]) for row in rows]
        finally:
            conn.close()

    def get_all_embeddings(self) -> list[tuple[int, np.ndarray]]:
        """Get all embeddings with person IDs for active persons.

        Returns:
            List of (person_id, embedding) tuples.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT e.person_id, e.embedding
                FROM embeddings e
                JOIN persons p ON e.person_id = p.id
                WHERE p.is_active = 1
            """)
            rows = cursor.fetchall()
            return [(row["person_id"], bytes_to_embedding(row["embedding"])) for row in rows]
        finally:
            conn.close()

    def delete_person(self, person_id: int) -> bool:
        """Hard delete person and all embeddings (GDPR compliance).

        Args:
            person_id: ID of the person to delete.

        Returns:
            True if person was deleted, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            if deleted:
                logger.info("Deleted person id=%d", person_id)
            return deleted
        finally:
            conn.close()

    def list_persons(self, active_only: bool = True) -> list[dict]:
        """List all registered persons.

        Args:
            active_only: If True, only return active persons.

        Returns:
            List of person dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT id, name, created_at, updated_at FROM persons"
            if active_only:
                query += " WHERE is_active = 1"
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def update_person_name(self, person_id: int, new_name: str) -> bool:
        """Update a person's name.

        Args:
            person_id: ID of the person.
            new_name: New name to set.

        Returns:
            True if updated, False if person not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "UPDATE persons SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, datetime.now().isoformat(), person_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
