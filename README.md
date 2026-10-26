# airflow-posture-pipeline
A batch computer-vision pipeline orchestrated with Apache Airflow. It samples frames from videos, extracts human keypoints with YOLO pose, classifies posture (standing, sitting, lying, ...) with a geometry-based classifier, writes an evaluation report, and exports artifacts to AWS S3.

```
extract_keypoints -> classify_posture -> evaluate -> upload_to_s3
```

| Task | What it does |
|---|---|
| `extract_keypoints` | YOLO pose (CPU) on sampled frames of every video in `data/input` |
| `classify_posture` | Rule-based classifier on torso/leg angles, knee angles and gravity cues |
| `evaluate` | Label distribution, confidence, data-quality rates, latency (mean / p95) |
| `upload_to_s3` | Uploads run artifacts; skipped automatically if `S3_BUCKET` is unset |

Each run writes to `data/runs/<run_id>/` (`keypoints.json`, `predictions.json`, `report.json`).

## Quickstart

```bash
# 1. put one or more videos in data/input/
# 2. start Airflow (Postgres + scheduler + webserver)
docker compose build
docker compose up airflow-init
docker compose up -d
# 3. open http://localhost:8080  (admin / admin), unpause and trigger `posture_pipeline`
```

Optional S3 export: copy `.env.example` to `.env` and fill in the values.

## Develop and test

```bash
pip install "apache-airflow==2.9.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
pip install pytest ruff boto3
ruff check . && pytest -q
```

CI (GitHub Actions) runs lint, unit tests and a DAG-integrity check on every push and PR.

## Design notes

- The classifier is rule-based, so there is no training step. Evaluation measures data quality and latency rather than accuracy; labelled accuracy is a future extension.
- Heavy imports (`cv2`, `ultralytics`, `boto3`) are lazy so DAG parsing stays fast.
- Tasks pass file paths (not data) through XCom, so large payloads never hit the metadata DB.
- Runs on CPU only.

## License

MIT
