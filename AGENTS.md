# AGENTS.md

## Cursor Cloud specific instructions

### Overview
This is a Python 3.10 ML research pipeline (CCT-LSTM) for reproducing multimodal remote stress estimation on the UBFC-Phys dataset. It is a single-machine training pipeline with no web services, databases, or Docker containers.

### Environment
- Python 3.10 virtual environment at `.venv` (activate with `source .venv/bin/activate`)
- PyTorch CPU version is installed (no NVIDIA GPU available in Cloud VM); training scripts accept `--device cpu`
- All dependencies from `environment.yml` are installed via pip into the venv (not conda)
- `python3.10-tk` system package is required for `tkinter` (used by `preprocessing/integrity_and_masterManifest.py`)

### Running scripts
- Activate venv first: `source /workspace/.venv/bin/activate`
- All training scripts (`train_cct.py`, `train_cct_lstm.py`, `train_cct_lstm_levels.py`) accept `--help` for full argument list
- Use `--device cpu` when running on this VM (no GPU)
- The full data pipeline requires the external UBFC-Phys dataset which is not available in the Cloud VM; model/import verification and unit tests work without it

### Testing
- Unit tests: `cd scripts/dataset_visualization && python -m pytest test_data_mining.py -v`
- Smoke test (`smoke_test_data_mining.py`) requires the actual UBFC-Phys dataset
- Verify imports with the verification snippet in README section 1.2

### Key caveats
- The `environment.yml` specifies a CUDA 12.1 PyTorch wheel; in Cloud VM we use the CPU wheel instead (`torch==2.5.1+cpu`)
- `face_landmarker.task` is a 3.6MB binary tracked via Git LFS; ensure LFS is pulled if landmark extraction is needed
- No lint tools (flake8, ruff, etc.) are configured in this project
