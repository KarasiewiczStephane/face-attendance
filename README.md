# Face Recognition Attendance System

> Real-time attendance tracking using FaceNet embeddings with anti-spoofing and GDPR-compliant data handling.

## Features

- **Face Detection** -- MTCNN-based multi-task cascaded face detection with alignment
- **Face Recognition** -- FaceNet (InceptionResnetV1) producing 512-d embeddings with cosine similarity matching
- **Anti-Spoofing** -- Dual liveness detection via Eye Aspect Ratio (blink tracking) and LBP texture analysis
- **REST API** -- FastAPI endpoints for enrollment, verification, attendance queries, and reporting
- **Attendance Logging** -- Automatic deduplication, on-time/late/absent status, daily and weekly reports (JSON, CSV, Markdown)
- **Privacy Controls** -- GDPR-compliant: complete data deletion, configurable retention, full audit logging
- **Streamlit Dashboard** -- Interactive attendance visualization with log tables, daily charts, weekly summaries, and face database gallery
- **Webcam Demo** -- Real-time camera feed with overlaid bounding boxes, names, confidence scores, and liveness status

## Quick Start

```bash
# Clone and install
git clone git@github.com:KarasiewiczStephane/face-attendance.git
cd face-attendance
pip install -r requirements.txt

# 1. Run API server (http://localhost:8000)
make run

# 2. Launch the Streamlit dashboard (http://localhost:8501)
make dashboard

# 3. Run webcam demo (requires camera)
make demo
```

The dashboard uses synthetic demo data and does not require the API server to be running.
To enroll faces and log real attendance, start the API first and use the `/enroll` and `/verify` endpoints (see below).

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
  thresholds: [0.6, 0.7, 0.7]  # MTCNN stages
  device: "cpu"                 # or "cuda"

embedding:
  model: "vggface2"
  image_size: 160
  embedding_dim: 512

matching:
  similarity_threshold: 0.7
  algorithm: "cosine"

liveness:
  blink_threshold: 0.25     # eye aspect ratio
  texture_threshold: 0.5
  challenge_timeout: 10      # seconds

attendance:
  dedup_window_hours: 4
  work_start: "09:00"
  work_end: "18:00"
  late_threshold_minutes: 15

privacy:
  retention_days: 365
  audit_enabled: true

database:
  path: "data/attendance.db"

api:
  host: "0.0.0.0"
  port: 8000
```

## Project Structure

```
face-attendance/
├── src/
│   ├── api/               # FastAPI app, schemas, routes
│   ├── dashboard/         # Streamlit attendance visualization
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
