"""Pipeline steps: extract keypoints -> classify posture -> evaluate -> upload.

Each step reads/writes JSON files inside a per-run directory, so Airflow tasks
only need to pass file paths between each other.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

from posture.classifier import PostureClassifier
from posture.config import PostureLabel
from posture.keypoints import KeypointExtractor

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run_dir(data_dir: Path, run_id: str) -> Path:
    """Per-run output folder; run_id is sanitised because Airflow ids contain ':' and '+'."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    path = Path(data_dir) / "runs" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_keypoints(
    input_dir: Path,
    out_dir: Path,
    model_path: str = "yolo11n-pose.pt",
    frame_stride: int = 15,
    max_frames_per_video: int = 100,
    conf: float = 0.3,
) -> Path:
    """Run YOLO pose on sampled video frames and save raw keypoints."""
    import cv2  # heavy imports kept lazy so DAG parsing and unit tests stay fast
    from ultralytics import YOLO

    videos = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No video files found in {input_dir}. Add a video and re-run.")

    model = YOLO(model_path)
    records: List[dict] = []

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        frame_idx, used = 0, 0
        while used < max_frames_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                height, width = frame.shape[:2]
                start = time.perf_counter()
                result = model.predict(frame, conf=conf, verbose=False, device="cpu")[0]
                infer_ms = (time.perf_counter() - start) * 1000

                if result.keypoints is not None:
                    for person_id, kpts in enumerate(result.keypoints.data.cpu().numpy()):
                        records.append(
                            {
                                "video": video.name,
                                "frame": frame_idx,
                                "person": person_id,
                                "frame_height": height,
                                "frame_width": width,
                                "infer_ms": round(infer_ms, 2),
                                "keypoints": kpts.tolist(),  # 17 x [x, y, conf]
                            }
                        )
                used += 1
            frame_idx += 1
        cap.release()
        log.info("Processed %s: %d frames sampled", video.name, used)

    out_path = Path(out_dir) / "keypoints.json"
    out_path.write_text(json.dumps(records))
    log.info("Saved %d person detections to %s", len(records), out_path)
    return out_path


def classify_posture(keypoints_path: Path, out_dir: Path) -> Path:
    """Apply the rule-based posture classifier to every detection."""
    records = json.loads(Path(keypoints_path).read_text())
    extractor, classifier = KeypointExtractor(), PostureClassifier()

    predictions = []
    for rec in records:
        coords = [k[:2] for k in rec["keypoints"]]
        conf = [k[2] for k in rec["keypoints"]]
        kpts = extractor.extract(coords, conf)
        label, confidence = classifier.classify(kpts, rec["frame_height"])
        predictions.append(
            {
                "video": rec["video"],
                "frame": rec["frame"],
                "person": rec["person"],
                "label": label,
                "confidence": round(confidence, 3),
                "infer_ms": rec["infer_ms"],
            }
        )

    out_path = Path(out_dir) / "predictions.json"
    out_path.write_text(json.dumps(predictions))
    return out_path


def evaluate(predictions_path: Path, out_dir: Path) -> Path:
    """Summarise a run: label mix, confidence, data quality and latency."""
    preds = json.loads(Path(predictions_path).read_text())
    n = len(preds)

    labels = Counter(p["label"] for p in preds)
    latencies = [p["infer_ms"] for p in preds]
    confidences = [p["confidence"] for p in preds]

    report = {
        "num_detections": n,
        "label_distribution": dict(labels),
        "insufficient_keypoints_rate": round(labels[PostureLabel.INSUFFICIENT] / n, 3) if n else None,
        "unknown_rate": round(labels[PostureLabel.UNKNOWN] / n, 3) if n else None,
        "mean_confidence": round(statistics.mean(confidences), 3) if n else None,
        "mean_infer_ms": round(statistics.mean(latencies), 2) if n else None,
        "p95_infer_ms": round(statistics.quantiles(latencies, n=20)[-1], 2) if n >= 2 else None,
        "per_video": dict(Counter(p["video"] for p in preds)),
    }

    out_path = Path(out_dir) / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Evaluation report: %s", report)
    return out_path


def upload_to_s3(run_path: Path, bucket: Optional[str], prefix: str = "posture-pipeline") -> List[str]:
    """Upload run artifacts to S3. Skips cleanly when no bucket is configured."""
    if not bucket:
        log.warning("S3_BUCKET not set - skipping upload (artifacts stay in %s)", run_path)
        return []

    import boto3

    s3 = boto3.client("s3")
    uploaded = []
    for file in Path(run_path).glob("*.json"):
        key = f"{prefix}/{Path(run_path).name}/{file.name}"
        s3.upload_file(str(file), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    log.info("Uploaded %d artifacts", len(uploaded))
    return uploaded
