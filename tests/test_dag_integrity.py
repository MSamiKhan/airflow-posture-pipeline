"""DAG integrity: it must import cleanly and have the expected shape."""

from pathlib import Path

import pytest

pytest.importorskip("airflow", reason="Airflow runs in CI/Docker, not on native Windows")

from airflow.models import DagBag  # noqa: E402

DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"


def test_no_import_errors():
    bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert bag.import_errors == {}


def test_posture_dag_structure():
    bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    dag = bag.get_dag("posture_pipeline")
    assert dag is not None
    assert {t.task_id for t in dag.tasks} == {
        "extract_keypoints", "classify_posture", "evaluate", "upload_to_s3",
    }
    assert dag.get_task("extract_keypoints").downstream_task_ids == {"classify_posture"}
    assert dag.get_task("classify_posture").downstream_task_ids == {"evaluate"}
    assert dag.get_task("evaluate").downstream_task_ids == {"upload_to_s3"}
