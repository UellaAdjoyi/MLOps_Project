from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
from importlib.metadata import version
import torch

def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata.

    TODO: collect at least:
      - Git commit SHA and whether the working tree is dirty;
      - identity of the prepared training + validation manifests;
      - Python, PyTorch, torchvision, DVC and MLflow versions;
      - OS/platform and CUDA runtime (when available);
      - compute device; random seed; image size; training parameters;
      - MLflow Run ID.

    Hints: subprocess, platform, importlib.metadata, torch,
    training_data_identity().
    """
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        git_dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
            .strip()
            .splitlines()
        )
    except subprocess.CalledProcessError:
        git_sha = "unavailable"
        git_dirty = []

    return {
        "git": {
            "commit_sha": git_sha,
            "dirty": bool(git_dirty),
        },
        "data": {
            "training_data_identity": training_data_identity(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "torchvision": version("torchvision"),
            "dvc": version("dvc"),
            "mlflow": version("mlflow"),
            "platform": platform.platform(),
            "cuda": (
                torch.version.cuda
                if torch.cuda.is_available()
                else None
            ),
        },
        "configuration": {
            "image_size": params.get("data", {}).get("image_size"),
            "batch_size": params.get("train", {}).get("batch_size"),
            "epochs": params.get("train", {}).get("epochs"),
            "learning_rate": params.get("train", {}).get("learning_rate"),
            "dropout": params.get("train", {}).get("dropout"),
            "augmentation_enabled": params.get("augmentation", {}).get("enabled"),
            "horizontal_flip": params.get("augmentation", {}).get("horizontal_flip"),
            "rotation_degrees": params.get("augmentation", {}).get("rotation_degrees"),
        },
        "execution": {
            "device": device,
            "random_state": params.get("train", {}).get("random_state"),
            "mlflow_run_id": mlflow_run_id,
        },
    }

def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
