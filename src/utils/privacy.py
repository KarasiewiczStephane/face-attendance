"""Privacy controls for GDPR-style data management.

Implements audit logging, complete person deletion, data retention
policies, and data export for privacy compliance.
"""

import sqlite3
from datetime import datetime, timedelta

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class AuditLogger:
    """Log all data access for privacy compliance.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory.

        Returns:
            SQLite connection with Row factory.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log(
        self,
        action: str,
        person_id: int | None = None,
        details: str | None = None,
        ip_address: str | None = None,
    ) -> int:
        """Log an audit event.

        Args:
            action: Action type (enroll, verify, delete, export, view, update).
            person_id: Associated person ID (optional).
            details: Human-readable details.
            ip_address: Client IP address.

        Returns:
            ID of the created audit log entry.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO audit_log (action, person_id, details, ip_address)
                VALUES (?, ?, ?, ?)
            """,
                (action, person_id, details, ip_address),
            )
            conn.commit()
            logger.debug("Audit log: action=%s, person_id=%s", action, person_id)
            return cursor.lastrowid
        finally:
            conn.close()

    def get_audit_log(
        self,
        person_id: int | None = None,
        action: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit log with optional filters.

        Args:
            person_id: Filter by person ID.
            action: Filter by action type.
            start_date: Filter entries after this date.
            end_date: Filter entries before this date.
            limit: Maximum number of entries to return.

        Returns:
            List of audit log entry dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM audit_log WHERE 1=1"
            params: list = []

            if person_id is not None:
                query += " AND person_id = ?"
                params.append(person_id)
            if action is not None:
                query += " AND action = ?"
                params.append(action)
            if start_date is not None:
                query += " AND timestamp >= ?"
                params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
            if end_date is not None:
                query += " AND timestamp <= ?"
                params.append(end_date.strftime("%Y-%m-%d %H:%M:%S"))

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()


class PrivacyManager:
    """Manage data retention and deletion for GDPR compliance.

    Args:
        db_path: Path to the SQLite database file.
        retention_days: Number of days to retain data.
    """

    def __init__(self, db_path: str, retention_days: int = 365) -> None:
        self.db_path = db_path
        self.retention_days = retention_days
        self.audit = AuditLogger(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory.

        Returns:
            SQLite connection with Row factory.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def delete_person_completely(
        self,
        person_id: int,
        requester_ip: str | None = None,
    ) -> dict:
        """GDPR-style complete deletion of a person's data.

        Removes person record, all embeddings (cascade), attendance
        records, and anonymizes audit log entries.

        Args:
            person_id: ID of the person to delete.
            requester_ip: IP of the requester for audit.

        Returns:
            Dict with success status and deletion details.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT name FROM persons WHERE id = ?", (person_id,))
            person = cursor.fetchone()
            if not person:
                return {"success": False, "error": "Person not found"}

            cursor.execute("PRAGMA foreign_keys = ON")

            cursor.execute("DELETE FROM attendance WHERE person_id = ?", (person_id,))
            attendance_deleted = cursor.rowcount

            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))

            cursor.execute(
                """
                UPDATE audit_log
                SET details = '[DELETED]'
                WHERE person_id = ?
            """,
                (person_id,),
            )

            conn.commit()

            self.audit.log(
                action="delete",
                person_id=person_id,
                details=f"Complete deletion of person: {person['name']}",
                ip_address=requester_ip,
            )

            logger.info("Deleted person id=%d completely", person_id)
            return {
                "success": True,
                "person_id": person_id,
                "attendance_records_deleted": attendance_deleted,
            }
        except Exception as e:
            conn.rollback()
            logger.error("Failed to delete person id=%d: %s", person_id, e)
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def apply_retention_policy(self) -> dict:
        """Delete data older than retention period.

        Returns:
            Dict with counts of deleted records.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

        try:
            cursor.execute("DELETE FROM embeddings WHERE created_at < ?", (cutoff_str,))
            embeddings_deleted = cursor.rowcount

            cursor.execute("DELETE FROM attendance WHERE timestamp < ?", (cutoff_str,))
            attendance_deleted = cursor.rowcount

            audit_cutoff = datetime.now() - timedelta(days=self.retention_days * 2)
            cursor.execute(
                "DELETE FROM audit_log WHERE timestamp < ?",
                (audit_cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            audit_deleted = cursor.rowcount

            conn.commit()

            self.audit.log(
                action="retention_cleanup",
                details=(
                    f"Deleted: {embeddings_deleted} embeddings, "
                    f"{attendance_deleted} attendance, {audit_deleted} audit"
                ),
            )

            logger.info(
                "Retention policy applied: %d embeddings, %d attendance, %d audit deleted",
                embeddings_deleted,
                attendance_deleted,
                audit_deleted,
            )
            return {
                "embeddings_deleted": embeddings_deleted,
                "attendance_deleted": attendance_deleted,
                "audit_deleted": audit_deleted,
            }
        finally:
            conn.close()

    def export_person_data(self, person_id: int) -> dict:
        """Export all data for a person (GDPR data portability).

        Args:
            person_id: ID of the person.

        Returns:
            Dict with all person data or error.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
            person = cursor.fetchone()
            if not person:
                return {"error": "Person not found"}

            cursor.execute(
                "SELECT id, model_version, created_at FROM embeddings WHERE person_id = ?",
                (person_id,),
            )
            embeddings = [dict(row) for row in cursor.fetchall()]

            cursor.execute("SELECT * FROM attendance WHERE person_id = ?", (person_id,))
            attendance = [dict(row) for row in cursor.fetchall()]

            cursor.execute("SELECT * FROM audit_log WHERE person_id = ?", (person_id,))
            audit = [dict(row) for row in cursor.fetchall()]

            self.audit.log(
                action="export",
                person_id=person_id,
                details="Data export requested",
            )

            return {
                "person": dict(person),
                "embeddings": embeddings,
                "attendance": attendance,
                "audit_log": audit,
                "exported_at": datetime.now().isoformat(),
            }
        finally:
            conn.close()
