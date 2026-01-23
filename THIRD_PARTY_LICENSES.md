# Third-Party Licenses

This project uses the following third-party libraries, frameworks, and model assets. We gratefully acknowledge their creators and maintainers.

---

## Model Assets

### MediaPipe Face Landmarker

The `face_landmarker.task` model file included in this repository is provided by Google.

- **Source**: [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- **Download URL**: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- **License**: Content is licensed under [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Usage**: Used for facial landmark extraction from video frames

---

## Python Dependencies

### pyVHR (Reference Script Only)

- **Repository**: https://github.com/phuselab/pyVHR
- **License**: GPL-3.0
- **Note**: The file `preprocessing/rppg_extractor.py` references pyVHR for rPPG signal extraction. This script is included **for documentation and provenance purposes only** and is **not imported or executed** within the main pipeline of this repository. The actual pipeline consumes pre-exported rPPG JSON files. pyVHR and its dependencies are NOT managed by this project's `environment.yml`.

### MediaPipe

- **Repository**: https://github.com/google-ai-edge/mediapipe
- **License**: Apache-2.0
- **PyPI**: https://pypi.org/project/mediapipe/
- **Usage**: Facial landmark detection via the FaceLandmarker API

### pyts

- **Repository**: https://github.com/johannfaouzi/pyts
- **License**: BSD-3-Clause
- **PyPI**: https://pypi.org/project/pyts/
- **Usage**: Markov Transition Field (MTF) image generation from time series

### PyTorch

- **Repository**: https://github.com/pytorch/pytorch
- **License**: BSD-style (see https://github.com/pytorch/pytorch/blob/main/LICENSE)
- **Website**: https://pytorch.org/
- **Usage**: Deep learning framework for model training and inference

### TorchVision

- **Repository**: https://github.com/pytorch/vision
- **License**: BSD-3-Clause
- **Usage**: Image transforms and utilities

### OpenCV (opencv-contrib-python)

- **Repository**: https://github.com/opencv/opencv
- **License**: Apache-2.0 (with BSD-3-Clause for some modules)
- **PyPI**: https://pypi.org/project/opencv-contrib-python/
- **Usage**: Video/image I/O and processing

### scikit-learn

- **Repository**: https://github.com/scikit-learn/scikit-learn
- **License**: BSD-3-Clause
- **Usage**: PCA dimensionality reduction, cross-validation, evaluation metrics

### NumPy

- **Repository**: https://github.com/numpy/numpy
- **License**: BSD-3-Clause
- **Usage**: Numerical array operations

### Pillow

- **Repository**: https://github.com/python-pillow/Pillow
- **License**: HPND (Historical Permission Notice and Disclaimer)
- **Usage**: Image loading and manipulation

### pandas

- **Repository**: https://github.com/pandas-dev/pandas
- **License**: BSD-3-Clause
- **Usage**: Data manipulation and CSV handling

### tqdm

- **Repository**: https://github.com/tqdm/tqdm
- **License**: MIT / MPL-2.0
- **Usage**: Progress bar display

### Weights & Biases (wandb)

- **Repository**: https://github.com/wandb/wandb
- **License**: MIT
- **Usage**: Experiment tracking (optional)

### vit-pytorch

- **Repository**: https://github.com/lucidrains/vit-pytorch
- **License**: MIT
- **Usage**: Vision Transformer implementation reference

### NeuroKit2

- **Repository**: https://github.com/neuropsychology/NeuroKit
- **License**: MIT
- **Usage**: Physiological signal analysis (EDA visualization scripts)

### Seaborn

- **Repository**: https://github.com/mwaskom/seaborn
- **License**: BSD-3-Clause
- **Usage**: Statistical data visualization

### SciPy

- **Repository**: https://github.com/scipy/scipy
- **License**: BSD-3-Clause
- **Usage**: Scientific computing utilities

---

## Acknowledgments

This project is a reproduction/implementation of the CCT-LSTM architecture described in:

> Ouzar, Y., Bousefsaf, F., Maaoui, C., & Pruski, A. (2024). Video-based multimodal spontaneous emotion recognition using facial expressions and physiological signals. *WACV 2024*.

We thank the original authors for their work and the UBFC-Phys dataset maintainers for making the data publicly available.
