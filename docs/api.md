# API Documentation

Base URL: `http://localhost:8000`

## Health Check

### `GET /health`

Check API health and component status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "model_loaded": true
}
```

## Enrollment

### `POST /enroll`

Register a new person with their face image.

**Query Parameters:**
- `name` (string, required): Person's name

**Body:** `multipart/form-data`
- `image` (file, required): Face image (JPEG/PNG)

**Example:**

```bash
curl -X POST "http://localhost:8000/enroll?name=Alice" \
  -F "image=@alice.jpg"
```

**Success Response (200):**

```json
{
  "success": true,
  "person_id": 1,
  "message": "Successfully enrolled Alice"
}
```

**Failure Responses:**

```json
{"success": false, "message": "No face detected in image"}
{"success": false, "message": "Liveness check failed - please use a live camera"}
```

## Verification

### `POST /verify`

Verify a face against enrolled persons and log attendance.

**Body:** `multipart/form-data`
- `image` (file, required): Face image (JPEG/PNG)

**Example:**

```bash
curl -X POST "http://localhost:8000/verify" -F "image=@photo.jpg"
```

**Success Response (200):**

```json
{
  "success": true,
  "person_id": 1,
  "name": "Alice",
  "confidence": 0.95,
  "liveness_score": 0.87,
  "attendance_logged": true,
  "message": "Attendance logged"
}
```

**Failure Responses:**

```json
{"success": false, "message": "No face detected"}
{"success": false, "liveness_score": 0.2, "message": "Liveness check failed"}
{"success": false, "liveness_score": 0.9, "message": "Face not recognized"}
```

## Attendance

### `GET /attendance/today`

Get today's attendance list with summary.

**Example:**

```bash
curl http://localhost:8000/attendance/today
```

**Response (200):**

```json
{
  "date": "2025-01-15",
  "records": [
    {
      "id": 1,
      "person_id": 1,
      "name": "Alice",
      "timestamp": "2025-01-15 08:50:00",
      "confidence": 0.95,
      "liveness_score": 0.87,
      "status": "present"
    }
  ],
  "summary": {
    "total": 1,
    "present": 1,
    "late": 0
  }
}
```

### `GET /attendance/report`

Get attendance report for a date range.

**Query Parameters:**
- `start` (string, required): Start date (YYYY-MM-DD)
- `end` (string, required): End date (YYYY-MM-DD)
- `format` (string, optional): Output format: `json` (default), `csv`, or `markdown`

**Examples:**

```bash
# JSON format
curl "http://localhost:8000/attendance/report?start=2025-01-01&end=2025-01-31"

# CSV format
curl "http://localhost:8000/attendance/report?start=2025-01-01&end=2025-01-31&format=csv"

# Markdown format
curl "http://localhost:8000/attendance/report?start=2025-01-01&end=2025-01-31&format=markdown"
```

## Person Management

### `GET /persons`

List all registered persons.

**Response (200):**

```json
{
  "persons": [
    {"id": 1, "name": "Alice", "embedding_count": 1}
  ]
}
```

### `DELETE /person/{person_id}`

Delete a person and all their data (GDPR-compliant).

**Example:**

```bash
curl -X DELETE http://localhost:8000/person/1
```

**Success Response (200):**

```json
{
  "success": true,
  "message": "Person deleted successfully",
  "records_deleted": 5
}
```

**Error Response (404):**

```json
{"detail": "Person not found"}
```

## Audit Log

### `GET /audit`

Get audit log entries.

**Query Parameters:**
- `person_id` (int, optional): Filter by person ID
- `action` (string, optional): Filter by action type (`enroll`, `verify`, `delete`)
- `limit` (int, optional): Max entries (default: 100, max: 1000)

**Example:**

```bash
curl "http://localhost:8000/audit?limit=10"
```

**Response (200):**

```json
{
  "entries": [
    {
      "id": 1,
      "action": "enroll",
      "person_id": 1,
      "details": "Enrolled: Alice",
      "ip_address": "127.0.0.1",
      "timestamp": "2025-01-15 08:45:00"
    }
  ]
}
```
