# Face Recognition Attendance System

> Real-time attendance tracking using FaceNet embeddings with anti-spoofing and GDPR-compliant data handling.

## Features

- **Face Detection** -- MTCNN-based multi-task cascaded face detection with alignment
- **Face Recognition** -- FaceNet (InceptionResnetV1) producing 512-d embeddings with cosine similarity matching
- **Anti-Spoofing** -- Dual liveness detection via Eye Aspect Ratio (blink tracking) and LBP texture analysis
- **REST API** -- FastAPI endpoints for enrollment, verification, attendance queries, and reporting
- **Attendance Logging** -- Automatic deduplication, on-time/late/absent status, daily and weekly reports (JSON, CSV, Markdown)
- **Privacy Controls** -- GDPR-compliant: complete data deletion, configurable retention, full audit logging
- **Webcam Demo** -- Real-time camera feed with overlaid bounding boxes, names, confidence scores, and liveness status

## Quick Start

```bash
# Clone
git clone git@github.com:KarasiewiczStephane/face-attendance.git
cd face-attendance

# Install
pip install -r requirements.txt

# Run API server
make run
# API available at http://localhost:8000

# Run webcam demo
make demo
```

## Docker

```bash
# Build and run
docker compose up -d

# Or manually
make docker-build
make docker-run
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and component status |
| `POST` | `/enroll?name=Alice` | Register a new person with face image |
| `POST` | `/verify` | Verify face, check liveness, log attendance |
| `GET` | `/attendance/today` | Get today's attendance list with summary |
| `GET` | `/attendance/report?start=2025-01-01&end=2025-01-31&format=json` | Date range report (json, csv, markdown) |
| `DELETE` | `/person/{id}` | Delete person and all data (GDPR) |
| `GET` | `/persons` | List all registered persons |
| `GET` | `/audit` | Get audit log entries |

### Example: Enroll a Person

```bash
curl -X POST "http://localhost:8000/enroll?name=Alice" \
  -F "image=@photo.jpg"
```

### Example: Verify and Log Attendance

```bash
curl -X POST "http://localhost:8000/verify" \
  -F "image=@photo.jpg"
```

### Example: Get Today's Attendance

```bash
curl http://localhost:8000/attendance/today
```

Full API documentation: [docs/api.md](docs/api.md)

## Architecture

```
                    +-----------+
                    |  Client   |
                    | (Browser/ |
                    |   curl)   |
                    +-----+-----+
                          |
                    +-----v-----+
                    | FastAPI   |
                    | REST API  |
                    +-----+-----+
                          |
          +---------------+---------------+
          |               |               |
    +-----v-----+  +------v------+  +-----v-----+
    |   Face     |  |  Matching   |  | Attendance|
    | Processor  |  |  Service    |  |  Database |
    +-----+------+  +------+------+  +-----+-----+
          |                |               |
    +-----v-----+   +------v------+  +-----v-----+
    |  MTCNN    |   | Face        |  | SQLite    |
    |  Detector |   | Database    |  | (reports, |
    +-----+-----+   +------+------+  |  privacy) |
          |                |          +-----------+
    +-----v-----+   +------v------+
    |  FaceNet  |   | Liveness    |
    | Embedder  |   | Detector    |
    +-----------+   +-------------+
```

See [docs/architecture.md](docs/architecture.md) for detailed component documentation.

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
face_detection:
  min_face_size: 20
  device: "cpu"              # or "cuda"

matching:
  similarity_threshold: 0.7  # cosine similarity cutoff

liveness:
  blink_threshold: 0.25      # EAR threshold for blink detection
  texture_threshold: 0.5     # LBP entropy threshold

attendance:
  dedup_window_hours: 4      # prevent duplicate logs
  work_start: "09:00"
  late_threshold_minutes: 15

privacy:
  retention_days: 365        # auto-delete after N days
```

## Project Structure

```
face-attendance/
├── src/
│   ├── api/               # FastAPI app, schemas, routes
│   ├── database/          # SQLite schema, face DB, attendance DB
│   ├── detection/         # MTCNN detector, FaceNet embedder, liveness
│   ├── matching/          # Cosine similarity matcher, caching service
│   ├── reporting/         # CSV/Markdown report generation
│   ├── demo/              # Webcam demo mode
│   └── utils/             # Config, logging, privacy/audit
├── tests/                 # Unit and integration tests
├── configs/               # YAML configuration
├── .github/workflows/     # CI/CD pipeline
├── Dockerfile             # Multi-stage production build
├── docker-compose.yml     # Container orchestration
├── Makefile               # Dev commands
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Ruff, pytest, coverage config
```

## Privacy and GDPR

- **No raw images stored** -- only 512-dimensional face embeddings
- **Complete deletion** -- `DELETE /person/{id}` removes all person data, embeddings, and attendance records
- **Configurable retention** -- automatic cleanup of data older than the configured retention period
- **Audit logging** -- every enrollment, verification, and deletion is logged with timestamp and client IP
- **Data export** -- export all data for a specific person on request

## Development

```bash
# Run linter
make lint

# Run tests with coverage
make test

# Clean caches
make clean
```

## License

MIT
