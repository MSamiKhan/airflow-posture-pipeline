"""
### Posture pipeline

extract_keypoints -> classify_posture -> evaluate -> upload_to_s3

Runs YOLO pose on sampled frames of videos in `data/input`, classifies posture
with a geometry-based classifier, writes an evaluation report, and uploads the
artifacts to S3 (skipped if `S3_BUCKET` is not set).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

DATA_DIR = Path(os.getenv("PIPELINE_DATA_DIR", "/opt/airflow/data"))
MODEL_PATH = os.getenv("YOLO_MODEL", "yolo11n-pose.pt")
S3_BUCKET = os.getenv("S3_BUCKET", "")


@dag(
    dag_id="posture_pipeline",
    description="Batch posture classification from video, with evaluation and S3 export",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["computer-vision", "posture"],
    doc_md=__doc__,
)
def posture_pipeline():
    @task
    def extract_keypoints() -> str:
        from posture import pipeline

        out_dir = pipeline.run_dir(DATA_DIR, get_current_context()["run_id"])
        path = pipeline.extract_keypoints(DATA_DIR / "input", out_dir, model_path=MODEL_PATH)
        return str(path)

    @task
    def classify_posture(keypoints_path: str) -> str:
        from posture import pipeline

        path = pipeline.classify_posture(Path(keypoints_path), Path(keypoints_path).parent)
        return str(path)

    @task
    def evaluate(predictions_path: str) -> str:
        from posture import pipeline

        path = pipeline.evaluate(Path(predictions_path), Path(predictions_path).parent)
        return str(path)

    @task
    def upload_to_s3(report_path: str) -> list[str]:
        from posture import pipeline

        return pipeline.upload_to_s3(Path(report_path).parent, S3_BUCKET)

    upload_to_s3(evaluate(classify_posture(extract_keypoints())))


posture_pipeline()
