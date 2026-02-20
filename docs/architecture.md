# Architecture

## System Overview

The Face Recognition Attendance System processes face images through a pipeline of detection, embedding generation, liveness verification, identity matching, and attendance logging.

## Components

### Face Detection (`src/detection/face_detector.py`)

Uses MTCNN (Multi-task Cascaded Convolutional Networks) for face detection and alignment. Returns bounding boxes and detection probabilities. Configurable minimum face size and detection thresholds.

### Embedding Generation (`src/detection/embedding_generator.py`)

Uses InceptionResnetV1 (FaceNet) pretrained on VGGFace2 to generate 512-dimensional face embeddings. Supports batch processing and model versioning.

### Liveness Detection (`src/detection/liveness.py`)

Dual anti-spoofing approach:
- **Blink Detection**: Computes Eye Aspect Ratio (EAR) using dlib facial landmarks. Tracks blink history to distinguish live faces from static photos.
- **Texture Analysis**: Uses Local Binary Patterns (LBP) to compute texture entropy. Live faces produce higher entropy than printed photos or screen displays.

### Face Database (`src/database/face_db.py`)

SQLite-backed storage for person records and face embeddings. Embeddings are serialized as raw bytes for efficient storage and retrieval. Supports CRUD operations and person listing.

### Matching Service (`src/matching/__init__.py`)

Cosine similarity matching between query embeddings and enrolled faces. Includes an in-memory embedding cache that is invalidated on enrollment or deletion. Configurable similarity threshold (default: 0.7).

### Attendance Database (`src/database/attendance_db.py`)

Logs attendance events with deduplication within a configurable time window. Determines attendance status (present, late, absent) based on configured work hours. Supports date-range queries and daily summaries.

### Report Generator (`src/reporting/report_generator.py`)

Generates attendance reports in CSV and Markdown formats. Supports daily and weekly reports with person-level breakdowns. Reports can be saved to disk or returned as strings.

### Privacy Controls (`src/utils/privacy.py`)

GDPR-compliant privacy management:
- **AuditLogger**: Records all enrollment, verification, and deletion events with timestamps and client IPs.
- **PrivacyManager**: Handles complete person deletion (cascading to embeddings, attendance, audit), retention policy enforcement, and data export.

### REST API (`src/api/app.py`)

FastAPI application with lifespan-managed component initialization. Provides endpoints for enrollment, verification, attendance queries, reporting, person management, and audit access. Uses Pydantic schemas for request/response validation.

## Data Flow

### Enrollment

```
Image Upload -> MTCNN Detection -> FaceNet Embedding -> Liveness Check
  -> Store Person + Embedding in SQLite -> Invalidate Cache -> Audit Log
```

### Verification

```
Image Upload -> MTCNN Detection -> FaceNet Embedding -> Liveness Check
  -> Cosine Similarity Search (cached) -> Attendance Log (deduplicated)
  -> Audit Log -> Return Result
```

## Database Schema

```sql
persons(id, name, created_at)
embeddings(id, person_id, embedding, model_version, created_at)
attendance(id, person_id, timestamp, confidence, liveness_score, status)
audit_log(id, action, person_id, details, ip_address, timestamp)
```

## Security Considerations

- No raw face images are stored; only 512-d embedding vectors
- Liveness detection prevents photo-based spoofing attacks
- All data access operations are audit-logged
- Complete data deletion available on request (GDPR Article 17)
- Configurable automatic data retention policies
- CORS middleware configured for API access control
