"""Attendance report generation in CSV and Markdown formats.

Generates daily and weekly attendance reports with present/absent/late
statistics, attendance rates, and exportable formats.
"""

import csv
import io
from datetime import datetime, timedelta
from pathlib import Path

from ..database.attendance_db import AttendanceDatabase
from ..database.face_db import FaceDatabase
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ReportGenerator:
    """Generate attendance reports in CSV and Markdown formats.

    Args:
        attendance_db: AttendanceDatabase instance.
        face_db: FaceDatabase instance.
    """

    def __init__(self, attendance_db: AttendanceDatabase, face_db: FaceDatabase) -> None:
        self.attendance_db = attendance_db
        self.face_db = face_db

    def generate_daily_csv(self, date: datetime) -> str:
        """Generate CSV report for a single day.

        Args:
            date: Date to report on.

        Returns:
            CSV-formatted string.
        """
        summary = self.attendance_db.get_daily_summary(date)
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Attendance Report", summary["date"]])
        writer.writerow([])
        writer.writerow(["Name", "Status", "Check-in Time"])

        for person in summary["present"]:
            writer.writerow([person["name"], "Present", person["time"]])

        for person in summary["late"]:
            writer.writerow([person["name"], "Late", person["time"]])

        for person in summary["absent"]:
            writer.writerow([person["name"], "Absent", "-"])

        writer.writerow([])
        writer.writerow(["Summary"])
        s = summary["summary"]
        writer.writerow(["Total", s["total"]])
        writer.writerow(["Present", s["present"]])
        writer.writerow(["Late", s["late"]])
        writer.writerow(["Absent", s["absent"]])
        total = s["total"]
        rate = (s["present"] + s["late"]) / max(total, 1) * 100
        writer.writerow(["Attendance Rate", f"{rate:.1f}%"])

        return output.getvalue()

    def generate_daily_markdown(self, date: datetime) -> str:
        """Generate Markdown report for a single day.

        Args:
            date: Date to report on.

        Returns:
            Markdown-formatted string.
        """
        summary = self.attendance_db.get_daily_summary(date)
        s = summary["summary"]
        total = s["total"]
        rate = (s["present"] + s["late"]) / max(total, 1) * 100

        lines = [
            f"# Attendance Report: {summary['date']}",
            "",
            "## Summary",
            f"- **Total Employees:** {total}",
            f"- **Present:** {s['present']}",
            f"- **Late:** {s['late']}",
            f"- **Absent:** {s['absent']}",
            f"- **Attendance Rate:** {rate:.1f}%",
            "",
            "## Details",
            "",
            "| Name | Status | Check-in Time |",
            "|------|--------|---------------|",
        ]

        for person in summary["present"]:
            lines.append(f"| {person['name']} | Present | {person['time']} |")

        for person in summary["late"]:
            lines.append(f"| {person['name']} | Late | {person['time']} |")

        for person in summary["absent"]:
            lines.append(f"| {person['name']} | Absent | - |")

        return "\n".join(lines)

    def generate_weekly_csv(
        self,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> str:
        """Generate CSV report for a week.

        Args:
            start_date: Start of the reporting period.
            end_date: End of the reporting period (default: start + 6 days).

        Returns:
            CSV-formatted string.
        """
        if end_date is None:
            end_date = start_date + timedelta(days=6)

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Weekly Attendance Report"])
        writer.writerow([f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"])
        writer.writerow([])

        persons = self.face_db.list_persons()
        dates = self._date_range(start_date, end_date)

        header = ["Name"] + [d.strftime("%a %m/%d") for d in dates] + ["Total Days", "Attendance %"]
        writer.writerow(header)

        for person in persons:
            row = [person["name"]]
            days_present = 0

            for date in dates:
                summary = self.attendance_db.get_daily_summary(date)
                present_ids = [p["id"] for p in summary["present"] + summary["late"]]

                if person["id"] in present_ids:
                    is_late = person["id"] in [p["id"] for p in summary["late"]]
                    row.append("L" if is_late else "P")
                    days_present += 1
                else:
                    row.append("A")

            row.append(str(days_present))
            row.append(f"{days_present / len(dates) * 100:.1f}%")
            writer.writerow(row)

        return output.getvalue()

    def generate_weekly_markdown(
        self,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> str:
        """Generate Markdown report for a week.

        Args:
            start_date: Start of the reporting period.
            end_date: End of the reporting period (default: start + 6 days).

        Returns:
            Markdown-formatted string.
        """
        if end_date is None:
            end_date = start_date + timedelta(days=6)

        persons = self.face_db.list_persons()
        dates = self._date_range(start_date, end_date)

        lines = [
            "# Weekly Attendance Report",
            f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "",
            "## Attendance Matrix",
            "",
        ]

        header = "| Name | " + " | ".join([d.strftime("%a") for d in dates]) + " | Total | % |"
        separator = "|" + "|".join(["------"] * (len(dates) + 3)) + "|"
        lines.append(header)
        lines.append(separator)

        for person in persons:
            row = f"| {person['name']} "
            days_present = 0

            for date in dates:
                summary = self.attendance_db.get_daily_summary(date)
                present_ids = [p["id"] for p in summary["present"]]
                late_ids = [p["id"] for p in summary["late"]]

                if person["id"] in present_ids:
                    row += "| P "
                    days_present += 1
                elif person["id"] in late_ids:
                    row += "| L "
                    days_present += 1
                else:
                    row += "| A "

            pct = days_present / len(dates) * 100
            row += f"| {days_present} | {pct:.0f}% |"
            lines.append(row)

        return "\n".join(lines)

    def save_report(self, content: str, filepath: Path) -> None:
        """Save report content to a file.

        Args:
            content: Report content string.
            filepath: Output file path.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        logger.info("Report saved to %s", filepath)

    @staticmethod
    def _date_range(start: datetime, end: datetime) -> list[datetime]:
        """Generate list of dates between start and end inclusive.

        Args:
            start: Start date.
            end: End date.

        Returns:
            List of datetime objects.
        """
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        return dates
