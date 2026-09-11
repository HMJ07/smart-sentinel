import sqlite3
import cv2
import os
from datetime import datetime

class EventLogger:
    def __init__(self, db_path="events.db", captures_dir="captures"):
        self.db_path = db_path
        self.captures_dir = captures_dir
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
                    image_path TEXT
                )
            ''')
            conn.commit()

    def log_event(self, event_type, description, frame=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_path = ""

        if frame is not None:
            filename = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            img_path = os.path.join(self.captures_dir, filename)
            cv2.imwrite(img_path, frame)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (timestamp, event_type, description, image_path) VALUES (?, ?, ?, ?)",
                (timestamp, event_type, description, img_path)
            )
            conn.commit()
