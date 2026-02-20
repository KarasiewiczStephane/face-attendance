"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel


class EnrollResponse(BaseModel):
    """Response for person enrollment."""

    success: bool
    person_id: int | None = None
    message: str


class VerifyResponse(BaseModel):
    """Response for face verification."""

    success: bool
    person_id: int | None = None
    name: str | None = None
    confidence: float | None = None
    liveness_score: float | None = None
    attendance_logged: bool = False
    message: str


class AttendanceRecord(BaseModel):
    """Single attendance record."""

    id: int
    person_id: int
    name: str
    timestamp: str
    confidence: float
    liveness_score: float | None = None
    status: str


class AttendanceSummary(BaseModel):
    """Attendance count summary."""

    total: int
    present: int
    late: int
    absent: int


class AttendanceListResponse(BaseModel):
    """Response for attendance listing."""

    date: str
    records: list[AttendanceRecord]
    summary: AttendanceSummary


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str
    version: str
    database: str
    model_loaded: bool


class DeleteResponse(BaseModel):
    """Response for person deletion."""

    success: bool
    message: str
    records_deleted: int | None = None


class PersonResponse(BaseModel):
    """Person information response."""

    id: int
    name: str
    created_at: str


class AuditLogEntry(BaseModel):
    """Single audit log entry."""

    id: int
    action: str
    person_id: int | None = None
    details: str | None = None
    timestamp: str
    ip_address: str | None = None


class ExportDataResponse(BaseModel):
    """Response for person data export."""

    person: dict
    embeddings: list[dict]
    attendance: list[dict]
    audit_log: list[dict]
    exported_at: str
