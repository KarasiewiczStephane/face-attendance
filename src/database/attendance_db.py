"""Attendance logging with time-window deduplication.

Manages attendance records in SQLite with configurable deduplication
windows, late arrival detection, and daily summary generation.
"""

import sqlite3
from datetime import datetime, timedelta

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class AttendanceDatabase:
    """SQLite-backed attendance logging with deduplication.

    Args:
        db_path: Path to the SQLite database file.
        dedup_hours: Time window in hours for deduplication.
    """

    def __init__(self, db_path: str, dedup_hours: float = 4.0) -> None:
        self.db_path = db_path
        self.dedup_window = timedelta(hours=dedup_hours)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory.

        Returns:
            SQLite connection with Row factory.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log_attendance(
        self,
        person_id: int,
        confidence: float,
        liveness_score: float | None = None,
        status: str = "present",
    ) -> int | None:
        """Log attendance with deduplication.

        Args:
            person_id: ID of the recognized person.
            confidence: Face matching confidence score.
            liveness_score: Liveness detection score.
            status: Attendance status ('present', 'late', 'early_departure').

        Returns:
            attendance_id if logged, None if deduplicated.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cutoff = datetime.now() - self.dedup_window
            cursor.execute(
                """
                SELECT id FROM attendance
                WHERE person_id = ? AND timestamp > ?
                ORDER BY timestamp DESC LIMIT 1
            """,
                (person_id, cutoff.strftime("%Y-%m-%d %H:%M:%S")),
            )

            if cursor.fetchone():
                logger.debug("Attendance deduplicated for person_id=%d", person_id)
                return None

            cursor.execute(
                """
                INSERT INTO attendance (person_id, confidence, liveness_score, status)
                VALUES (?, ?, ?, ?)
            """,
                (person_id, confidence, liveness_score, status),
            )

            conn.commit()
            logger.info("Attendance logged for person_id=%d, status=%s", person_id, status)
            return cursor.lastrowid
        finally:
            conn.close()

    def determine_status(
        self,
        timestamp: datetime,
        work_start: str = "09:00",
        late_threshold_minutes: int = 15,
    ) -> str:
        """Determine attendance status based on check-in time.

        Args:
            timestamp: Check-in time.
            work_start: Expected work start time (HH:MM).
            late_threshold_minutes: Grace period in minutes.

        Returns:
            Status string: 'present' or 'late'.
        """
        work_start_time = datetime.strptime(work_start, "%H:%M").time()
        late_cutoff = (
            datetime.combine(datetime.today(), work_start_time)
            + timedelta(minutes=late_threshold_minutes)
        ).time()

        check_time = timestamp.time()

        if check_time <= late_cutoff:
            return "present"
        return "late"

    def get_attendance_by_date(self, date: datetime) -> list[dict]:
        """Get all attendance records for a specific date.

        Args:
            date: Date to query.

        Returns:
            List of attendance record dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            date_str = date.strftime("%Y-%m-%d")
            cursor.execute(
                """
                SELECT a.*, p.name
                FROM attendance a
                JOIN persons p ON a.person_id = p.id
                WHERE date(a.timestamp) = ?
                ORDER BY a.timestamp
            """,
                (date_str,),
            )

            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_attendance_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Get attendance records for a date range.

        Args:
            start_date: Start of range (inclusive).
            end_date: End of range (inclusive).

        Returns:
            List of attendance record dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT a.*, p.name
                FROM attendance a
                JOIN persons p ON a.person_id = p.id
                WHERE date(a.timestamp) BETWEEN ? AND ?
                ORDER BY a.timestamp
            """,
                (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
            )

            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_daily_summary(self, date: datetime) -> dict:
        """Get attendance summary for a specific date.

        Args:
            date: Date to summarize.

        Returns:
            Dict with date, present/late/absent lists, and summary counts.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            date_str = date.strftime("%Y-%m-%d")

            cursor.execute("SELECT id, name FROM persons WHERE is_active = 1")
            all_persons = {row["id"]: row["name"] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT person_id, status, MIN(timestamp) as first_check_in
                FROM attendance
                WHERE date(timestamp) = ?
                GROUP BY person_id
            """,
                (date_str,),
            )
            attendance = {row["person_id"]: dict(row) for row in cursor.fetchall()}

            present = []
            late = []
            absent = []

            for person_id, name in all_persons.items():
                if person_id in attendance:
                    record = attendance[person_id]
                    entry = {
                        "id": person_id,
                        "name": name,
                        "time": record["first_check_in"],
                    }
                    if record["status"] == "late":
                        late.append(entry)
                    else:
                        present.append(entry)
                else:
                    absent.append({"id": person_id, "name": name})

            return {
                "date": date_str,
                "present": present,
                "late": late,
                "absent": absent,
                "summary": {
                    "total": len(all_persons),
                    "present": len(present),
                    "late": len(late),
                    "absent": len(absent),
                },
            }
        finally:
            conn.close()
