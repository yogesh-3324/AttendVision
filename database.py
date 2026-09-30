"""
database.py — SQLite persistence using raw sqlite3.
Stores: students, attendance_sessions, attendance_records.
Face embeddings are stored as numpy .npy files on disk (faster than BLOB for arrays).
"""

import sqlite3
import json
import os
from datetime import datetime
from config import DB_PATH


def _conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id          TEXT PRIMARY KEY,
            roll_number TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            class_name  TEXT NOT NULL,
            section     TEXT,
            phone       TEXT,
            photo_path  TEXT,
            photos_count INTEGER DEFAULT 0,
            is_active   INTEGER DEFAULT 1,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id              TEXT PRIMARY KEY,
            class_name      TEXT NOT NULL,
            session_date    TEXT NOT NULL,
            subject         TEXT,
            teacher_name    TEXT,
            image_path      TEXT,
            total_detected  INTEGER DEFAULT 0,
            total_recognized INTEGER DEFAULT 0,
            total_absent    INTEGER DEFAULT 0,
            emails_sent     INTEGER DEFAULT 0,
            status          TEXT DEFAULT 'done',
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS attendance_records (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            student_id      TEXT,
            status          TEXT NOT NULL,
            confidence      REAL,
            email_sent      INTEGER DEFAULT 0,
            FOREIGN KEY(session_id) REFERENCES attendance_sessions(id),
            FOREIGN KEY(student_id) REFERENCES students(id)
        );
        """)


# ── Students ─────────────────────────────────────────────────────────────────

def add_student(student_id, roll_number, name, email, class_name,
                section=None, phone=None, photo_path=None, photos_count=0):
    with _conn() as conn:
        conn.execute(
            """INSERT INTO students
               (id,roll_number,name,email,class_name,section,phone,photo_path,photos_count)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (student_id, roll_number, name, email, class_name,
             section, phone, photo_path, photos_count),
        )


def update_student_photos(student_id, photos_count):
    with _conn() as conn:
        conn.execute(
            "UPDATE students SET photos_count=? WHERE id=?",
            (photos_count, student_id),
        )


def get_student_by_roll(roll_number):
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE roll_number=? AND is_active=1", (roll_number,)
        ).fetchone()
        return dict(row) if row else None


def get_student_by_id(student_id):
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE id=?", (student_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_students(class_name=None):
    with _conn() as conn:
        if class_name:
            rows = conn.execute(
                "SELECT * FROM students WHERE is_active=1 AND class_name=? ORDER BY roll_number",
                (class_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM students WHERE is_active=1 ORDER BY class_name, roll_number"
            ).fetchall()
        return [dict(r) for r in rows]


def get_all_classes():
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT class_name FROM students WHERE is_active=1 ORDER BY class_name"
        ).fetchall()
        return [r["class_name"] for r in rows]


def delete_student(student_id):
    with _conn() as conn:
        conn.execute("UPDATE students SET is_active=0 WHERE id=?", (student_id,))


# ── Attendance ────────────────────────────────────────────────────────────────

def save_session(session_id, class_name, session_date, subject, teacher_name,
                 image_path, total_detected, total_recognized, total_absent):
    with _conn() as conn:
        conn.execute(
            """INSERT INTO attendance_sessions
               (id,class_name,session_date,subject,teacher_name,image_path,
                total_detected,total_recognized,total_absent)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (session_id, class_name, session_date, subject, teacher_name,
             image_path, total_detected, total_recognized, total_absent),
        )


def save_record(record_id, session_id, student_id, status, confidence):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO attendance_records (id,session_id,student_id,status,confidence) VALUES (?,?,?,?,?)",
            (record_id, session_id, student_id, status, confidence),
        )


def mark_emails_sent(session_id):
    with _conn() as conn:
        conn.execute("UPDATE attendance_sessions SET emails_sent=1 WHERE id=?", (session_id,))
        conn.execute("UPDATE attendance_records SET email_sent=1 WHERE session_id=?", (session_id,))


def get_session(session_id):
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM attendance_sessions WHERE id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def get_session_records(session_id):
    """Returns records joined with student info."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT r.*, s.name, s.roll_number, s.email, s.class_name
               FROM attendance_records r
               LEFT JOIN students s ON s.id = r.student_id
               WHERE r.session_id = ?
               ORDER BY s.roll_number""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_sessions(class_name=None, limit=20):
    with _conn() as conn:
        if class_name:
            rows = conn.execute(
                "SELECT * FROM attendance_sessions WHERE class_name=? ORDER BY session_date DESC LIMIT ?",
                (class_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM attendance_sessions ORDER BY session_date DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_student_attendance_history(student_id, limit=50):
    with _conn() as conn:
        rows = conn.execute(
            """SELECT r.status, r.confidence, s.session_date, s.subject, s.class_name
               FROM attendance_records r
               JOIN attendance_sessions s ON s.id = r.session_id
               WHERE r.student_id = ?
               ORDER BY s.session_date DESC LIMIT ?""",
            (student_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_student_attendance_percent(student_id, class_name=None):
    with _conn() as conn:
        q = """SELECT
               COUNT(*) as total,
               SUM(CASE WHEN r.status='present' THEN 1 ELSE 0 END) as present
               FROM attendance_records r
               JOIN attendance_sessions s ON s.id = r.session_id
               WHERE r.student_id = ?"""
        params = [student_id]
        if class_name:
            q += " AND s.class_name = ?"
            params.append(class_name)
        row = conn.execute(q, params).fetchone()
        if row and row["total"] > 0:
            return round(row["present"] / row["total"] * 100, 1)
        return None


# Initialise on import
init_db()
