"""FastAPI application for the face attendance system.

Provides REST endpoints for face enrollment, verification,
attendance queries, privacy controls, and health checks.
"""

import io
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image

from ..database.attendance_db import AttendanceDatabase
from ..database.face_db import FaceDatabase
from ..detection import FaceProcessor
from ..detection.liveness import LivenessDetector
from ..matching import MatchingService
from ..reporting.report_generator import ReportGenerator
from ..utils.config import get_settings
from ..utils.logger import setup_logger
from ..utils.privacy import AuditLogger, PrivacyManager
from .schemas import (
    AttendanceListResponse,
    AttendanceRecord,
    AttendanceSummary,
    DeleteResponse,
    EnrollResponse,
    HealthResponse,
    VerifyResponse,
)

logger = setup_logger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up application components."""
    settings = get_settings()
    config = settings.load_config()

    db_path = config["database"]["path"]

    _state["face_processor"] = FaceProcessor(config)
    _state["face_db"] = FaceDatabase(db_path)
    _state["attendance_db"] = AttendanceDatabase(
        db_path,
        dedup_hours=config["attendance"]["dedup_window_hours"],
    )
    _state["matching_service"] = MatchingService(
        _state["face_db"],
        threshold=config["matching"]["similarity_threshold"],
    )
    _state["liveness_detector"] = LivenessDetector(
        blink_threshold=config["liveness"]["blink_threshold"],
        texture_threshold=config["liveness"]["texture_threshold"],
    )
    _state["privacy_manager"] = PrivacyManager(
        db_path,
        retention_days=config["privacy"]["retention_days"],
    )
    _state["report_generator"] = ReportGenerator(
        _state["attendance_db"],
        _state["face_db"],
    )
    _state["config"] = config

    logger.info("Application started")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="Face Attendance API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    return request.client.host if request.client else "unknown"


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and component status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database="connected",
        model_loaded="face_processor" in _state,
    )


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(
    request: Request,
    name: str,
    image: UploadFile = File(...),  # noqa: B008
) -> EnrollResponse:
    """Register a new person with their face image."""
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        face_processor: FaceProcessor = _state["face_processor"]
        embedding, bbox, prob = face_processor.process_image(pil_image)

        if embedding is None:
            return EnrollResponse(success=False, message="No face detected in image")

        liveness_detector: LivenessDetector = _state["liveness_detector"]
        img_array = np.array(pil_image)
        liveness = liveness_detector.check_liveness(img_array)

        if not liveness["overall_live"]:
            return EnrollResponse(
                success=False,
                message="Liveness check failed - please use a live camera",
            )

        face_db: FaceDatabase = _state["face_db"]
        model_version = face_processor.embedder.get_model_version()
        person_id = face_db.register_person(name, [embedding], model_version)

        matching_service: MatchingService = _state["matching_service"]
        matching_service.invalidate_cache()

        AuditLogger(face_db.db_path).log(
            "enroll", person_id, f"Enrolled: {name}", _get_client_ip(request)
        )

        return EnrollResponse(
            success=True,
            person_id=person_id,
            message=f"Successfully enrolled {name}",
        )
    except Exception as e:
        logger.error("Enrollment failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    request: Request,
    image: UploadFile = File(...),  # noqa: B008
) -> VerifyResponse:
    """Verify face and log attendance."""
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        face_processor: FaceProcessor = _state["face_processor"]
        embedding, bbox, prob = face_processor.process_image(pil_image)

        if embedding is None:
            return VerifyResponse(success=False, message="No face detected")

        liveness_detector: LivenessDetector = _state["liveness_detector"]
        img_array = np.array(pil_image)
        liveness = liveness_detector.check_liveness(img_array)

        if not liveness["overall_live"]:
            return VerifyResponse(
                success=False,
                liveness_score=liveness["confidence"],
                message="Liveness check failed",
            )

        matching_service: MatchingService = _state["matching_service"]
        match = matching_service.identify(embedding)

        if not match:
            return VerifyResponse(
                success=False,
                liveness_score=liveness["confidence"],
                message="Face not recognized",
            )

        person, confidence = match

        attendance_db: AttendanceDatabase = _state["attendance_db"]
        config = _state["config"]
        status = attendance_db.determine_status(
            datetime.now(),
            work_start=config["attendance"]["work_start"],
            late_threshold_minutes=config["attendance"]["late_threshold_minutes"],
        )
        attendance_id = attendance_db.log_attendance(
            person["id"],
            confidence,
            liveness["confidence"],
            status,
        )

        face_db: FaceDatabase = _state["face_db"]
        AuditLogger(face_db.db_path).log(
            "verify",
            person["id"],
            f"Verified: {person['name']}, confidence: {confidence:.2f}",
            _get_client_ip(request),
        )

        return VerifyResponse(
            success=True,
            person_id=person["id"],
            name=person["name"],
            confidence=confidence,
            liveness_score=liveness["confidence"],
            attendance_logged=attendance_id is not None,
            message="Attendance logged" if attendance_id else "Already logged recently",
        )
    except Exception as e:
        logger.error("Verification failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/attendance/today", response_model=AttendanceListResponse)
async def get_today_attendance() -> AttendanceListResponse:
    """Get today's attendance list."""
    attendance_db: AttendanceDatabase = _state["attendance_db"]
    summary = attendance_db.get_daily_summary(datetime.now())
    records = attendance_db.get_attendance_by_date(datetime.now())

    attendance_records = [
        AttendanceRecord(
            id=r["id"],
            person_id=r["person_id"],
            name=r["name"],
            timestamp=r["timestamp"],
            confidence=r["confidence"],
            liveness_score=r.get("liveness_score"),
            status=r["status"],
        )
        for r in records
    ]

    return AttendanceListResponse(
        date=summary["date"],
        records=attendance_records,
        summary=AttendanceSummary(**summary["summary"]),
    )


@app.get("/attendance/report")
async def get_attendance_report(
    start: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end: str = Query(..., description="End date (YYYY-MM-DD)"),
    format: str = Query("json", description="Output format: json, csv, or markdown"),
):
    """Get attendance report for a date range."""
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    report_gen: ReportGenerator = _state["report_generator"]
    attendance_db: AttendanceDatabase = _state["attendance_db"]

    if format == "csv":
        content = report_gen.generate_weekly_csv(start_date, end_date)
        return PlainTextResponse(content, media_type="text/csv")
    elif format == "markdown":
        content = report_gen.generate_weekly_markdown(start_date, end_date)
        return PlainTextResponse(content, media_type="text/markdown")
    else:
        records = attendance_db.get_attendance_range(start_date, end_date)
        return {"start": start, "end": end, "records": records}


@app.delete("/person/{person_id}", response_model=DeleteResponse)
async def delete_person(person_id: int, request: Request) -> DeleteResponse:
    """Delete a person and all their data (GDPR)."""
    privacy_manager: PrivacyManager = _state["privacy_manager"]
    result = privacy_manager.delete_person_completely(person_id, _get_client_ip(request))

    if result["success"]:
        matching_service: MatchingService = _state["matching_service"]
        matching_service.invalidate_cache()
        return DeleteResponse(
            success=True,
            message="Person deleted successfully",
            records_deleted=result.get("attendance_records_deleted"),
        )
    raise HTTPException(status_code=404, detail=result.get("error"))


@app.get("/audit")
async def get_audit_log(
    person_id: int | None = None,
    action: str | None = None,
    limit: int = Query(100, le=1000),
):
    """Get audit log entries."""
    face_db: FaceDatabase = _state["face_db"]
    audit = AuditLogger(face_db.db_path)
    entries = audit.get_audit_log(person_id=person_id, action=action, limit=limit)
    return {"entries": entries}


@app.get("/persons")
async def list_persons():
    """List all registered persons."""
    face_db: FaceDatabase = _state["face_db"]
    persons = face_db.list_persons()
    return {"persons": persons}
