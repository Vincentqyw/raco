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
- Returns: `view0`, `view1`, `H_0to1`, `seq_name`, `is_illumination`, `image_size`

### Training (`train.py`)

Three-stage training:
1. **detector**: Train keypoint detection (policy gradient loss)
2. **ranker**: Train ranking scores (soft ranking loss)
3. **covariance**: Train covariance estimation

Features:
- TensorBoard logging (`outputs/tb_logs/`)
- Periodic evaluation on HPatches during training
- Checkpoint saving every `save_interval` steps

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
| `raco/models/utils/losses.py` | Detector, Ranker, Covariance losses |
| `raco/datasets/oxford_paris.py` | Training dataset with homography sampling |
| `raco/datasets/hpatches.py` | Evaluation dataset (HPatches protocol) |
| `raco/geometry/homography.py` | Homography utilities, warping, error metrics |
| `train.py` | Training script with eval and TensorBoard |
| `eval.py` | Standalone evaluation with metrics |
| `configs/default.yaml` | OmegaConf configuration |

## Glue-Factory Integration

This codebase adopts patterns from [glue-factory](https://github.com/cvg/glue-factory):
- **BaseModel/BaseDataset**: Unified interfaces with `default_conf`
- **Factory functions**: `get_model()`, `get_dataset()` for dynamic loading
- **Config-driven**: OmegaConf for all configuration
- **Data format**: `{"view0": {"image": tensor}, "view1": {"image": tensor}, "H_0to1": tensor}`

Reference glue-factory code is in `thirdparty/glue-factory/` for consultation only - do not import directly.
