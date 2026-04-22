import sqlite3
from datetime import datetime
from pathlib import Path
import config as cfg


def get_conn():
    conn = sqlite3.connect(cfg.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                duration_s INTEGER NOT NULL,
                cat_detected INTEGER NOT NULL,
                cat_detected_drinking INTEGER NOT NULL,
                occupancy_score REAL,
                positive_frames INTEGER,
                snapshot_path TEXT
            );

            CREATE TABLE IF NOT EXISTS current_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_update TEXT,
                cat_detected INTEGER DEFAULT 0,
                cat_detected_drinking INTEGER DEFAULT 0,
                event_active INTEGER DEFAULT 0,
                current_event_id INTEGER DEFAULT 0,
                occupancy_score REAL DEFAULT 0,
                positive_frames INTEGER DEFAULT 0,
                event_start_time TEXT,
                latest_snapshot TEXT DEFAULT '',
                live_feed_enabled INTEGER DEFAULT 0,
                latest_live_frame TEXT DEFAULT ''
            );
            """
        )
        conn.execute("INSERT OR IGNORE INTO current_status (id) VALUES (1)")


def update_status(
    cat_detected=0,
    cat_detected_drinking=0,
    event_active=0,
    current_event_id=0,
    occupancy_score=0.0,
    positive_frames=0,
    event_start_time=None,
    latest_snapshot="",
):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE current_status
            SET last_update = ?,
                cat_detected = ?,
                cat_detected_drinking = ?,
                event_active = ?,
                current_event_id = ?,
                occupancy_score = ?,
                positive_frames = ?,
                event_start_time = ?,
                latest_snapshot = ?
            WHERE id = 1
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                int(cat_detected),
                int(cat_detected_drinking),
                int(event_active),
                int(current_event_id),
                float(occupancy_score),
                int(positive_frames),
                event_start_time,
                latest_snapshot,
            ),
        )


def insert_event(
    event_id,
    started_at,
    ended_at,
    duration_s,
    cat_detected,
    cat_detected_drinking,
    occupancy_score,
    positive_frames,
    snapshot_path,
):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO events (
                event_id, started_at, ended_at, duration_s,
                cat_detected, cat_detected_drinking,
                occupancy_score, positive_frames, snapshot_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(event_id),
                started_at,
                ended_at,
                int(duration_s),
                int(cat_detected),
                int(cat_detected_drinking),
                float(occupancy_score),
                int(positive_frames),
                snapshot_path,
            ),
        )


def set_live_feed_enabled(enabled: bool):
    with get_conn() as conn:
        conn.execute(
            "UPDATE current_status SET live_feed_enabled = ? WHERE id = 1",
            (1 if enabled else 0,),
        )


def get_live_feed_enabled() -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT live_feed_enabled FROM current_status WHERE id = 1"
        ).fetchone()
    return bool(row[0]) if row else False


def update_live_frame(path: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE current_status SET latest_live_frame = ? WHERE id = 1",
            (path,),
        )


def reset_all_data(delete_images: bool = True):
    with get_conn() as conn:
        conn.execute("DELETE FROM events")
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name = 'events'"
        )
        conn.execute(
            """
            UPDATE current_status
            SET last_update = ?,
                cat_detected = 0,
                cat_detected_drinking = 0,
                event_active = 0,
                current_event_id = 0,
                occupancy_score = 0,
                positive_frames = 0,
                event_start_time = NULL,
                latest_snapshot = '',
                live_feed_enabled = 0,
                latest_live_frame = ''
            WHERE id = 1
            """,
            (datetime.now().isoformat(timespec="seconds"),),
        )

    if delete_images:
        for folder in (cfg.SNAPSHOT_DIR, cfg.LIVE_DIR):
            folder_path = Path(folder)
            for item in folder_path.glob("*.jpg"):
                try:
                    item.unlink()
                except OSError:
                    pass
