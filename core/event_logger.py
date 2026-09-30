import sqlite3
import cv2
import os
from datetime import datetime

class EventLogger:
    def __init__(self, db_path=None, captures_dir=None):
        from config.settings import Config
        self.db_path = db_path or Config.DB_PATH
        self.captures_dir = captures_dir or Config.CAPTURES_DIR
        os.makedirs(self.captures_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    image_path TEXT,
                    severity TEXT NOT NULL DEFAULT 'INFO'
                )
            ''')
            # Migración: bases de datos creadas antes de existir la columna severity
            cols = [r[1] for r in cursor.execute("PRAGMA table_info(events)")]
            if "severity" not in cols:
                cursor.execute("ALTER TABLE events ADD COLUMN severity TEXT NOT NULL DEFAULT 'INFO'")
            conn.commit()

    def log_event(self, event_type, description, frame=None, severity="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_path = ""

        if frame is not None:
            filename = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            img_path = os.path.join(self.captures_dir, filename)
            cv2.imwrite(img_path, frame)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (timestamp, event_type, description, image_path, severity) VALUES (?, ?, ?, ?, ?)",
                (timestamp, event_type, description, img_path, severity)
            )
            conn.commit()
