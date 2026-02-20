-- Enable foreign key support
PRAGMA foreign_keys = ON;

-- People/faces table
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Face embeddings (multiple per person)
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,  -- 512-d vector serialized as float32 bytes
    model_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);

-- Attendance records
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence REAL NOT NULL,
    liveness_score REAL,
    status TEXT DEFAULT 'present',  -- present, late, early_departure
    FOREIGN KEY (person_id) REFERENCES persons(id)
);

-- Audit log for privacy compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,  -- enroll, verify, delete, export
    person_id INTEGER,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_attendance_person_time ON attendance(person_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_audit_person ON audit_log(person_id);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(timestamp);
