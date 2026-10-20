"""Runs the API in-process on every usable sample and stores its responses, apart from the tape measurements."""
import datetime
import io
import json
import os
import subprocess

from measurement import ALGORITHM_VERSION

from .dataset import REPO_DIR


def code_revision():
    """Short git commit of the code being benchmarked, marked -dirty with uncommitted changes; None without git."""
    try:
        commit = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "-C", REPO_DIR, "status", "--porcelain", "--untracked-files=no"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if changes else "")


def predict(dataset, client=None):
    """One record per usable sample: the HTTP status and full JSON body the API returned."""
    if client is None:
        import app as app_module  # loads MediaPipe, so only when predictions are actually run

        client = app_module.app.test_client()
    revision = code_revision()
    records = []
    for sample in dataset.usable_samples():
        data = {"height_cm": f"{sample['height_cm_input']:g}"}
        for field, column in (("front", "front_image"), ("left_side", "side_image")):
            if sample[column] is not None:
                with open(dataset.image_path(sample[column]), "rb") as f:
                    data[field] = (io.BytesIO(f.read()), os.path.basename(sample[column]))
        response = client.post("/upload_images", data=data, content_type="multipart/form-data")
        body = response.get_json(silent=True)
        records.append({
            "sample_id": sample["sample_id"],
            "algorithm_version": ALGORITHM_VERSION,
            "code_revision": revision,
            "predicted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "http_status": response.status_code,
            "error_code": body.get("code") if response.status_code != 200 and body else None,
            "response": body,
        })
    return records


def predictions_path(dataset_path, version):
    return os.path.join(dataset_path, "predictions", f"{version}.jsonl")


def write_predictions(dataset_path, records, version=ALGORITHM_VERSION):
    """Replace the predictions file for this algorithm version."""
    path = predictions_path(dataset_path, version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_predictions(dataset_path, version):
    """{sample_id: record} for one algorithm version."""
    with open(predictions_path(dataset_path, version), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {record["sample_id"]: record for record in records}
