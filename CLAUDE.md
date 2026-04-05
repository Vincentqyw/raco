# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RaCo (Ranking and Covariance) is a neural network for learning robust and versatile keypoints in images. It provides:
- **Keypoint detection** with NMS and subpixel refinement
- **Ranking scores** indicating reliability for matching
- **Covariance estimates** (2x2 matrices) for spatial uncertainty

The codebase follows the **glue-factory** architecture pattern with modular datasets, models, and evaluation.

## Installation

```bash
pip install -e .
```

## Common Commands

```bash
# Training
python train.py --conf configs/default.yaml --stage detector
python train.py --conf configs/default.yaml --stage detector --eval_only

# Evaluation
python eval.py --conf configs/default.yaml --checkpoint <path> --scene_type all

# Lint and format
ruff check .
ruff format .
black .
isort .

# Type checking
mypy .
```

## Architecture

### Model (`raco/models/extractors/raco.py`)

The `RaCo` class implements `BaseModel` with:

1. **Encoder**: 4-level feature extraction using `ConvBlock` and `ResBlock` modules
2. **Feature Aggregation**: Multi-scale feature fusion (1x, 2x, 8x, 32x upsampled)
3. **Three prediction heads**:
   - **Score head**: Detection confidence via global softmax normalization
   - **Ranker head**: Predicts ranking scores for keypoint reliability
   - **Covariance head**: Outputs Cholesky factor elements → 2x2 covariance matrices

Key configurations (in `RaCo.default_conf`):
- `max_num_keypoints`: Maximum keypoints per image (default: 2048)
- `nms_radius`: NMS radius, must be odd (default: 3)
- `subpixel_sampling`: Enable subpixel refinement (default: True)
- `ranker`: Enable ranking module (default: True)
- `covariance_estimator`: Enable covariance prediction (default: True)
- `sort_by_ranker`: Sort keypoints by ranker scores (default: False)

### Datasets (glue-factory pattern)

All datasets inherit from `BaseDataset` in `raco/datasets/base_dataset.py`.

#### OxfordParisDataset (`raco/datasets/oxford_paris.py`)
Training dataset with corner-based homography sampling:
- Uses `sample_homography_corners()` from glue-factory with difficulty/translation/rotation
- Photometric augmentation (brightness, contrast, gamma, noise, blur)
- Config: `data_root`, `datasets`, `image_size`, `train_size`, `homography`, `photometric`

#### HPatchesDataset (`raco/datasets/hpatches.py`)
Evaluation dataset following SuperPoint/LoFTR protocol:
- Supports `ignore_large_scenes` for fair comparison (excludes 8 extreme resolution scenes)
- Scene types: `all`, `vantage` (v_*), `illumination` (i_*)
- Returns: `image0`, `image1`, `H_0to1`, `seq_name`, `is_illumination`, `image_size`

### Training

The training system is modularized into `raco/trainer/` module with `train.py` as the orchestration script.

#### Three-Stage Training

1. **detector**: Train keypoint detection (policy gradient loss)
2. **ranker**: Train ranking scores (soft ranking loss)
3. **covariance**: Train covariance estimation (negative log-likelihood)

#### Training Architecture (`raco/trainer/`)

- **`engine.py`**: `StageTrainer` class encapsulates all training logic for a single stage
  - Handles optimizer, scheduler, and loss function setup
  - Mixed precision training (AMP) support
  - Training loop with logging and evaluation
- **`losses.py`**: Stage-specific loss computation functions
  - `compute_detector_loss()`: Policy gradient with sparse keypoints
  - `compute_ranker_loss()`: Soft ranking with Spearman correlation
  - `compute_covariance_loss()`: Bidirectional covariance NLL
- **`metrics.py`**: TensorBoard logging utilities
  - Stage-specific metric logging
  - Gradient norm monitoring
  - Progress bar formatting
- **`checkpoint.py`**: Checkpoint save/load utilities
- **`mixed_precision.py`**: AMP setup utilities
- **`model_utils.py`**: Parameter freezing utilities

#### Key Features
- TensorBoard logging (`outputs/<timestamp>/tb_logs/`)
- Periodic evaluation on HPatches during training
- Checkpoint saving every `save_interval` steps
- Mixed precision training for faster GPU training
- Modular design for easy testing and extension

### Evaluation (`eval.py`)

Metrics computed (following glue-factory):
- **Repeatability**: @1px and @3px thresholds
- **Matching**: precision @1px/@3px, matching score
- **Homography estimation**: DLT and RANSAC-based
- **Corner error**: Mean corner displacement

Per-scene-type aggregation for illumination vs vantage scenes.

## Usage

### Inference

```python
import torch
from raco import RaCo
from raco.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"
extractor = RaCo().to(device).eval()

image = load_image("path/to/image.png").to(device)
output = extractor.extract(image)

# Output keys: keypoints, keypoint_scores, ranker_scores, covariances, image_size
```

### Training

```python
# From config
conf = OmegaConf.load("configs/default.yaml")
model = get_model("raco")(conf.model)
dataset = get_dataset("oxford_paris")(conf.dataset)
loader = dataset.get_data_loader("train")
```

## Key Files

| File | Description |
|------|-------------|
| `raco/models/extractors/raco.py` | Main RaCo model (BaseModel subclass) |
| `raco/models/losses/losses.py` | Detector, Ranker, Covariance loss classes |
| `raco/models/losses/soft_rank.py` | Soft ranking implementation (PAV algorithm) |
| `raco/datasets/oxford_paris.py` | Training dataset with homography sampling |
| `raco/datasets/hpatches.py` | Evaluation dataset (HPatches protocol) |
| `raco/geometry/homography.py` | Homography utilities, warping, error metrics |
| `raco/geometry/matching.py` | Keypoint matching utilities (MNN) |
| `raco/trainer/engine.py` | StageTrainer class - core training engine |
| `raco/trainer/losses.py` | Loss computation functions for each stage |
| `raco/trainer/metrics.py` | TensorBoard logging utilities |
| `raco/trainer/checkpoint.py` | Checkpoint save/load utilities |
| `raco/evaluation/evaluator.py` | Evaluation logic with metrics |
| `train.py` | Training orchestration script (~100 lines) |
| `eval.py` | Standalone evaluation with metrics |
| `configs/default.yaml` | OmegaConf configuration |

## Loss Functions (`raco/models/losses/`)

### DetectorLoss
Policy gradient loss following "Learning Feature Descriptors using Deep Neural Networks":
- **Reward**: Positive (+1.0) for inliers (d ≤ d_max), negative for outliers
- **Dynamic negative reward**: Increases over training steps
- **Sparse implementation**: Computes loss on sampled keypoints only
- **Reference**: Eq. 3 in RaCo paper

### RankingLoss
Soft ranking loss with two components:
- **Spearman loss**: MSE between matched keypoint ranks (Eq. 4)
- **Pull loss**: Pull matched keypoints to rank 1, unmatched to rank N (Eq. 5)
- Uses differentiable soft ranking via PAV algorithm (Pool Adjacent Violators)
- **Implementation**: `soft_rank.py` - pure Python/Numpy, no numba dependency

### CovarianceLoss
Negative log-likelihood loss for covariance estimation:
- **Bidirectional**: Computes loss in both directions (A→B and B→A)
- **Jacobian propagation**: Propagates covariance through homography
- **Mahalanobis distance**: Measures reprojection error with uncertainty
- **Reference**: Eq. 6-7 in RaCo paper

## Geometry Module (`raco/geometry/`)

### `homography.py`
Core homography operations:
- `transform_points_with_homography()`: Projects points with numerical stability
- `compute_homography_jacobian()`: Jacobian for covariance propagation
- Handles homogeneous coordinates with epsilon stability

### `matching.py`
Keypoint matching utilities:
- `compute_mutual_dist()`: Mutual Nearest Neighbors (MNN) matching
- `find_matches()`: Match keypoints between views using homography
- `get_valid_mask()`: Check if points within image bounds

## Glue-Factory Integration

This codebase adopts patterns from [glue-factory](https://github.com/cvg/glue-factory):
- **BaseModel/BaseDataset**: Unified interfaces with `default_conf`
- **Factory functions**: `get_model()`, `get_dataset()` for dynamic loading
- **Config-driven**: OmegaConf for all configuration
- **Data format**: `{"image0": {"image": tensor}, "image1": {"image": tensor}, "H_0to1": tensor}`

Reference glue-factory code is in `thirdparty/glue-factory/` for consultation only - do not import directly.

## Code Organization

The codebase follows modular design principles:

```
raco/
├── models/
│   ├── extractors/raco.py      # RaCo model architecture
│   ├── losses/                 # Loss functions
│   │   ├── losses.py           # DetectorLoss, RankingLoss, CovarianceLoss
│   │   └── soft_rank.py        # Differentiable soft ranking
│   └── utils/                  # Model utilities
├── datasets/                   # Dataset implementations
├── geometry/                   # Geometric utilities
│   ├── homography.py           # Homography operations
│   └── matching.py             # Keypoint matching
├── trainer/                    # Training engine and utilities
│   ├── engine.py               # StageTrainer class
│   ├── losses.py               # Loss computation functions
│   ├── metrics.py              # TensorBoard logging
│   ├── checkpoint.py           # Model checkpointing
│   ├── mixed_precision.py      # AMP utilities
│   └── model_utils.py          # Parameter management
├── evaluation/                 # Evaluation logic
│   └── evaluator.py            # run_eval function
└── utils/                      # General utilities
```

**Design Principles:**
- **Single Responsibility**: Each module has a clear, focused purpose
- **Separation of Concerns**: Training, evaluation, geometry, and losses are separate
- **Testability**: Pure functions and isolated components enable unit testing
- **Reusability**: Geometry and matching utilities can be used across training/eval/inference
