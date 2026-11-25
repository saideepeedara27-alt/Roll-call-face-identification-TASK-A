"""SQLite storage, unified binary embedding cache, and Wipe All Data support."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Iterator
import numpy as np

from rollcall.config import DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RollCallDB:
    """Manages local SQLite state for enrolled identities, embeddings, and corrections."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    identity_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reference_photos (
                    photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id TEXT NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
                    original_filename TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    crop_jpeg BLOB NOT NULL,
                    quality_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(identity_id, image_sha256, model_version)
                );

                CREATE TABLE IF NOT EXISTS embedding_cache (
                    image_sha256 TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    face_count INTEGER NOT NULL,
                    embedding BLOB,
                    crop_jpeg BLOB,
                    quality_json TEXT,
                    cached_at TEXT NOT NULL,
                    PRIMARY KEY (image_sha256, model_version)
                );

                CREATE TABLE IF NOT EXISTS session_corrections (
                    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_image_sha256 TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    previous_identity_id TEXT,
                    new_identity_id TEXT,
                    new_status TEXT NOT NULL,
                    action TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def clear_enrolled_roster(self) -> None:
        """Clear current enrolled identities and reference photos (preserving optional cache)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM reference_photos;")
            conn.execute("DELETE FROM identities;")

    def upsert_identity(self, identity_id: str, display_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO identities (identity_id, display_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(identity_id) DO UPDATE SET display_name = excluded.display_name;
                """,
                (identity_id, display_name, _utc_now()),
            )

    def add_reference_photo(
        self,
        identity_id: str,
        display_name: str,
        original_filename: str,
        image_sha256: str,
        model_version: str,
        embedding: np.ndarray,
        crop_jpeg: bytes,
        quality_dict: dict,
        persist_cache: bool = True,
    ) -> int:
        emb_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
        q_json = json.dumps(quality_dict)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO identities (identity_id, display_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(identity_id) DO UPDATE SET display_name = excluded.display_name;
                """,
                (identity_id, display_name, now),
            )
            cursor = conn.execute(
                """
                INSERT INTO reference_photos (
                    identity_id, original_filename, image_sha256, model_version,
                    embedding, crop_jpeg, quality_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identity_id, image_sha256, model_version) DO UPDATE SET
                    original_filename = excluded.original_filename,
                    embedding = excluded.embedding,
                    crop_jpeg = excluded.crop_jpeg,
                    quality_json = excluded.quality_json
                RETURNING photo_id;
                """,
                (
                    identity_id,
                    original_filename,
                    image_sha256,
                    model_version,
                    emb_bytes,
                    crop_jpeg,
                    q_json,
                    now,
                ),
            )
            photo_id = int(cursor.fetchone()["photo_id"])

            if persist_cache:
                conn.execute(
                    """
                    INSERT INTO embedding_cache (
                        image_sha256, model_version, face_count, embedding, crop_jpeg, quality_json, cached_at
                    )
                    VALUES (?, ?, 1, ?, ?, ?, ?)
                    ON CONFLICT(image_sha256, model_version) DO UPDATE SET
                        embedding = excluded.embedding,
                        crop_jpeg = excluded.crop_jpeg,
                        quality_json = excluded.quality_json,
                        cached_at = excluded.cached_at;
                    """,
                    (image_sha256, model_version, emb_bytes, crop_jpeg, q_json, now),
                )
            return photo_id

    def get_cached_single_face(
        self, image_sha256: str, model_version: str
    ) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT face_count, embedding, crop_jpeg, quality_json
                FROM embedding_cache
                WHERE image_sha256 = ? AND model_version = ?;
                """,
                (image_sha256, model_version),
            ).fetchone()
            if not row or row["embedding"] is None:
                return None
            return {
                "face_count": row["face_count"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32).copy(),
                "crop_jpeg": bytes(row["crop_jpeg"]),
                "quality": json.loads(row["quality_json"]),
            }

    def list_identities(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT i.identity_id, i.display_name, COUNT(r.photo_id) AS photo_count
                FROM identities i
                LEFT JOIN reference_photos r ON i.identity_id = r.identity_id
                GROUP BY i.identity_id
                ORDER BY i.display_name COLLATE NOCASE ASC;
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def load_all_exemplars(self, model_version: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.photo_id, r.identity_id, i.display_name, r.original_filename,
                       r.embedding, r.quality_json
                FROM reference_photos r
                JOIN identities i ON r.identity_id = i.identity_id
                WHERE r.model_version = ?
                ORDER BY i.display_name COLLATE NOCASE ASC;
                """,
                (model_version,),
            ).fetchall()
            result = []
            for r in rows:
                result.append(
                    {
                        "photo_id": r["photo_id"],
                        "identity_id": r["identity_id"],
                        "display_name": r["display_name"],
                        "original_filename": r["original_filename"],
                        "embedding": np.frombuffer(r["embedding"], dtype=np.float32).copy(),
                        "quality": json.loads(r["quality_json"]),
                    }
                )
            return result

    def get_reference_crop(self, photo_id: int) -> bytes | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT crop_jpeg FROM reference_photos WHERE photo_id = ?;",
                (photo_id,),
            ).fetchone()
            return bytes(row["crop_jpeg"]) if row else None

    def record_correction(
        self,
        group_image_sha256: str,
        face_index: int,
        previous_identity_id: str | None,
        new_identity_id: str | None,
        new_status: str,
        action: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_corrections (
                    group_image_sha256, face_index, previous_identity_id,
                    new_identity_id, new_status, action, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    group_image_sha256,
                    face_index,
                    previous_identity_id,
                    new_identity_id,
                    new_status,
                    action,
                    _utc_now(),
                ),
            )

    def wipe_all_data(self) -> dict:
        """Permanently delete all biometric embeddings, identities, and corrections."""
        with self._connect() as conn:
            counts = {
                "identities": conn.execute("SELECT COUNT(*) FROM identities;").fetchone()[0],
                "reference_photos": conn.execute("SELECT COUNT(*) FROM reference_photos;").fetchone()[0],
                "cached_embeddings": conn.execute("SELECT COUNT(*) FROM embedding_cache;").fetchone()[0],
                "corrections": conn.execute("SELECT COUNT(*) FROM session_corrections;").fetchone()[0],
            }
            conn.execute("DELETE FROM session_corrections;")
            conn.execute("DELETE FROM reference_photos;")
            conn.execute("DELETE FROM identities;")
            conn.execute("DELETE FROM embedding_cache;")
        return counts
