# Milestone 0

| Component | Issue | Impact on Reproducibility | Proposed Fix |
|-----------|--------|--------------------------|--------------|
| src/reproducibility.py | Random seed initialization is incomplete | Different executions may produce different results | Set seeds for random, NumPy, PyTorch and DataLoader workers |
| src/validate_data.py | Dataset validation logic is incomplete | Invalid data may enter the pipeline | Implement integrity and schema checks |
| src/provenance.py | MLflow provenance manifest is not generated | Experiment origin cannot be traced | Record dataset, git commit and environment information |
| src/train.py | MLflow logging is incomplete | Experiments cannot be compared accurately | Log parameters, metrics and artifacts |
| params.yaml | Parameters are not fully tracked | Changes may go unnoticed | Track all parameters through DVC and MLflow |
| DVC Pipeline | Pipeline stages are undefined | Workflow cannot be reproduced automatically | Define stages, dependencies and outputs |


Engineering checkpoint 1: A trained model file alone does not prove reproducibility because it does not contain the exact code version, dataset version, environment configuration, random seed or training parameters used to produce it.
