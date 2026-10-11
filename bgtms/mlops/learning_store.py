"""
BGTMS - Continuous Learning & Active Feedback Store (MLOps)
Automatically archives uploaded traffic photos & video keyframes to SQLite database
and Amazon S3 (s3://bgtms-ai-datasets/continuous_learning/) for autonomous model retraining.
"""

import os
import time
import json
import uuid
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class ContinuousLearningStore:
    """
    Manages continuous dataset ingestion, metadata logging in SQLite,
    and asynchronous cloud sync to Amazon S3.
    """

    def __init__(
        self,
        base_dir: str = "data/continuous_learning",
        db_path: Optional[str] = None,
        s3_bucket: str = "bgtms-ai-datasets",
        s3_prefix: str = "continuous_learning",
        aws_profile: str = "roadsense"
    ):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.labels_dir = self.base_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = Path(db_path) if db_path else self.base_dir / "traffic_learning.db"
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.aws_profile = aws_profile

        self._init_db()

    def _init_db(self):
        """Initializes SQLite tables for traffic captures and retraining history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    label_path TEXT NOT NULL,
                    s3_image_uri TEXT,
                    s3_label_uri TEXT,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    detections_count INTEGER NOT NULL,
                    total_pcu REAL NOT NULL,
                    class_counts TEXT NOT NULL,
                    detections_json TEXT NOT NULL,
                    is_hard_example INTEGER DEFAULT 0,
                    sync_s3_status TEXT DEFAULT 'pending',
                    retrained_status TEXT DEFAULT 'unprocessed'
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrain_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    samples_used INTEGER NOT NULL,
                    epochs INTEGER NOT NULL,
                    train_loss REAL NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    map_50 REAL NOT NULL,
                    status TEXT NOT NULL,
                    weights_promoted INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def record_capture(
        self,
        image: np.ndarray,
        filename: str,
        media_type: str,
        detections: List[Any],
        class_counts: Dict[str, int],
        total_pcu: float,
        async_s3_upload: bool = True
    ) -> Dict[str, Any]:
        """
        Saves image to disk, writes YOLO-format label file, inserts row into SQLite DB,
        and optionally kicks off asynchronous background S3 upload.
        """
        capture_id = str(uuid.uuid4())[:12]
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h, w = image.shape[:2]

        # 1. Save Image to disk
        local_img_path = self.images_dir / f"{capture_id}.jpg"
        cv2.imwrite(str(local_img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 2. Write YOLO label annotations: class_id xc yc w h (normalized)
        local_lbl_path = self.labels_dir / f"{capture_id}.txt"
        yolo_lines = []
        is_hard_example = 0

        for d in detections:
            # d can be BoundingBox object or dict
            if hasattr(d, "x1"):
                x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
                cls_id = d.class_id
                conf = d.confidence
            else:
                x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                cls_id = d.get("class_id", 0)
                conf = d.get("confidence", 0.5)

            # Flag as hard example if boundary confidence or distant small object
            bw = x2 - x1
            bh = y2 - y1
            if (0.24 <= conf <= 0.45) or (bw <= 25 and bh <= 25):
                is_hard_example = 1

            xc = ((x1 + x2) / 2.0) / max(1.0, float(w))
            yc = ((y1 + y2) / 2.0) / max(1.0, float(h))
            nw = max(0.001, bw / max(1.0, float(w)))
            nh = max(0.001, bh / max(1.0, float(h)))
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        with open(local_lbl_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(yolo_lines) + "\n")

        # Serialized detection list
        det_dicts = []
        for d in detections:
            if hasattr(d, "model_dump"):
                det_dicts.append(d.model_dump())
            elif isinstance(d, dict):
                det_dicts.append(d)
            else:
                det_dicts.append({
                    "x1": getattr(d, "x1", 0), "y1": getattr(d, "y1", 0),
                    "x2": getattr(d, "x2", 0), "y2": getattr(d, "y2", 0),
                    "class_name": getattr(d, "class_name", "car"),
                    "confidence": getattr(d, "confidence", 0.5)
                })

        s3_img_key = f"{self.s3_prefix}/images/{capture_id}.jpg"
        s3_lbl_key = f"{self.s3_prefix}/labels/{capture_id}.txt"
        s3_img_uri = f"s3://{self.s3_bucket}/{s3_img_key}"
        s3_lbl_uri = f"s3://{self.s3_bucket}/{s3_lbl_key}"

        # 3. Insert into SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO captures (
                    id, timestamp, media_type, original_filename,
                    image_path, label_path, s3_image_uri, s3_label_uri,
                    width, height, detections_count, total_pcu,
                    class_counts, detections_json, is_hard_example,
                    sync_s3_status, retrained_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'unprocessed')
            """, (
                capture_id, ts, media_type, filename,
                str(local_img_path), str(local_lbl_path), s3_img_uri, s3_lbl_uri,
                w, h, len(detections), total_pcu,
                json.dumps(class_counts), json.dumps(det_dicts), is_hard_example
            ))
            conn.commit()

        # 4. Trigger asynchronous S3 Upload in daemon thread
        if async_s3_upload and BOTO3_AVAILABLE:
            t = threading.Thread(
                target=self._upload_to_s3_background,
                args=(capture_id, str(local_img_path), s3_img_key, str(local_lbl_path), s3_lbl_key),
                daemon=True
            )
            t.start()

        return {
            "capture_id": capture_id,
            "timestamp": ts,
            "image_path": str(local_img_path),
            "label_path": str(local_lbl_path),
            "s3_uri": s3_img_uri,
            "is_hard_example": bool(is_hard_example),
            "detections_count": len(detections)
        }

    def _upload_to_s3_background(self, capture_id: str, img_path: str, img_key: str, lbl_path: str, lbl_key: str):
        """Asynchronously syncs captured traffic scene and annotations to S3."""
        try:
            session = boto3.Session(profile_name=self.aws_profile, region_name="ap-south-1")
            s3 = session.client("s3")
            s3.upload_file(img_path, self.s3_bucket, img_key)
            s3.upload_file(lbl_path, self.s3_bucket, lbl_key)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE captures SET sync_s3_status = 'synced' WHERE id = ?",
                    (capture_id,)
                )
                conn.commit()
        except Exception:
            # Fall back to default credentials if profile not available
            try:
                s3 = boto3.client("s3", region_name="ap-south-1")
                s3.upload_file(img_path, self.s3_bucket, img_key)
                s3.upload_file(lbl_path, self.s3_bucket, lbl_key)
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE captures SET sync_s3_status = 'synced' WHERE id = ?",
                        (capture_id,)
                    )
                    conn.commit()
            except Exception as e:
                # Local data remains safely preserved in SQLite and local disk
                pass

    def get_learning_stats(self) -> Dict[str, Any]:
        """Returns statistics on ingested data and retraining readiness."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(detections_count), SUM(total_pcu) FROM captures")
            row = cursor.fetchone()
            total_captures = row[0] or 0
            total_vehicles = row[1] or 0
            total_pcu = round(row[2] or 0.0, 1)

            cursor.execute("SELECT COUNT(*) FROM captures WHERE sync_s3_status = 'synced'")
            synced_s3 = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM captures WHERE is_hard_example = 1")
            hard_examples = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM captures WHERE retrained_status = 'unprocessed'")
            unprocessed = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*), MAX(timestamp) FROM retrain_runs")
            run_row = cursor.fetchone()
            retrain_runs = run_row[0] or 0
            last_retrain = run_row[1] or "Never"

            # Class counts aggregate
            cursor.execute("SELECT class_counts FROM captures ORDER BY timestamp DESC LIMIT 100")
            rows = cursor.fetchall()
            aggregate_classes = {}
            for r in rows:
                try:
                    c_dict = json.loads(r[0])
                    for k, v in c_dict.items():
                        aggregate_classes[k] = aggregate_classes.get(k, 0) + v
                except Exception:
                    pass

        return {
            "total_captures_in_db": total_captures,
            "total_vehicles_recorded": total_vehicles,
            "total_pcu_accumulated": total_pcu,
            "synced_to_s3_count": synced_s3,
            "hard_examples_flagged": hard_examples,
            "unprocessed_for_retraining": unprocessed,
            "retraining_threshold": 10,
            "ready_to_retrain": unprocessed >= 5,
            "retrain_runs_count": retrain_runs,
            "last_retrained_at": last_retrain,
            "class_distribution": aggregate_classes,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "storage_path": str(self.base_dir)
        }

    def get_unprocessed_samples(self) -> List[Tuple[Path, Path]]:
        """Returns pairs of (image_path, label_path) ready for training."""
        samples = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, image_path, label_path FROM captures WHERE retrained_status = 'unprocessed'"
            )
            rows = cursor.fetchall()
            for r in rows:
                img_p = Path(r[1])
                lbl_p = Path(r[2])
                if img_p.exists() and lbl_p.exists():
                    samples.append((img_p, lbl_p))
        return samples

    def mark_samples_trained(self, run_id: str, sample_ids: Optional[List[str]] = None):
        """Marks samples as trained in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if sample_ids:
                cursor.executemany(
                    "UPDATE captures SET retrained_status = 'trained' WHERE id = ?",
                    [(sid,) for sid in sample_ids]
                )
            else:
                cursor.execute(
                    "UPDATE captures SET retrained_status = 'trained' WHERE retrained_status = 'unprocessed'"
                )
            conn.commit()


# Singleton instance
learning_store = ContinuousLearningStore()
