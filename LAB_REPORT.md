# Milestone 0

| Component | Issue | Impact on Reproducibility | Proposed Fix |
|-----------|--------|--------------------------|--------------|
| src/reproducibility.py | Random seed initialization is incomplete | Different executions may produce different results | Set seeds for random, NumPy, PyTorch and DataLoader workers |
| src/validate_data.py | Dataset validation logic is incomplete | Invalid data may enter the pipeline | Implement integrity and schema checks |
| src/provenance.py | MLflow provenance manifest is not generated | Experiment origin cannot be traced | Record dataset, git commit and environment information |
| src/train.py | MLflow logging is incomplete | Experiments cannot be compared accurately | Log parameters, metrics and artifacts |
| params.yaml | Parameters are not fully tracked | Changes may go unnoticed | Track all parameters through DVC and MLflow |
| DVC Pipeline | Pipeline stages are undefined | Workflow cannot be reproduced automatically | Define stages, dependencies and outputs |


# Engineering checkpoint 1: 
 A trained model file alone does not prove reproducibility because it does not contain the exact code version, dataset version, environment configuration, random seed or training parameters used to produce it.


## Environment Files

### pyproject.toml

The `pyproject.toml` file defines the project's metadata, dependencies and development tools. It specifies which Python packages are required and their version constraints. This file serves as the main configuration file for the project and allows the environment to be recreated consistently.

### uv.lock

The `uv.lock` file locks the exact versions of all direct and transitive dependencies used by the project. Unlike pyproject.toml, which may specify version ranges, the lock file ensures that every developer installs exactly the same package versions, improving reproducibility across different machines.

### .venv/

The `.venv/` directory contains the local Python virtual environment created for the project. It isolates project dependencies from the system-wide Python installation, preventing conflicts with other projects and ensuring that the correct packages are used during development and testing.


# Engineering checkpoint 2:
random_state = 42 doesn't guarantee bit-for-bit results on every CPU/GPU environments beacause different hardware architectures, library versions, CUDA/CUdnn implementations and floating-point computations can produce small numerical differences even when the same seed is used

# Engineering checkpoint 3:
- The Git remote stores project files such as code, configuration, reports and DVC metadata files
- The DVC remote stores the large tracked data files, including the raw image dataset and other cached artifacts
- No, `git push` only transfers Git-tracked files

# Milestone 4 - Answers

## Q1
Yes. PREPARE depends on `reports/data_quality.json` because it must run after successful validation.

## Q2
Yes. PREPARE directly reads `data/raw`, so changes in raw data must trigger re-execution.

## Q3
TRAIN should track all parameters that affect preprocessing, augmentation, training, and reproducibility.

## Q4
EVALUATE must track `selection.run_id` because changing the selected MLflow run changes the evaluated model.

## Q5
Metrics:
- `reports/train_summary.json`
- `reports/test_metrics.json`

## Q6
Plots:
- `reports/training_curves.png`
- `reports/test_confusion_matrix.png`