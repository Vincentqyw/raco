"""HPatches dataset for evaluation."""

from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import torch

from .base_dataset import BaseDataset


# Large images that were ignored in previous papers (SuperPoint, LoFTR, etc.)
# These have extreme resolutions that cause memory issues or unfair comparison
IGNORED_SCENES = frozenset([
    "i_contruction",
    "i_crownnight",
    "i_dc",
    "i_pencils",
    "i_whitebuilding",
    "v_artisans",
    "v_astronautis",
    "v_talent",
])


def read_homography(path):
    """Read homography matrix from HPatches format.

    HPatches homography files may have variable formatting with double spaces.
    This handles those cases robustly.
    """
    with open(path) as f:
        result = []
        for line in f.readlines():
            # Remove double spaces
            while "  " in line:
                line = line.replace("  ", " ")
            line = line.replace(" \n", "").replace("\n", "")
            # Split and discard empty strings
            elements = list(filter(lambda s: s, line.split(" ")))
            if elements:
                result.append(elements)
        return np.array(result).astype(np.float32)


class HPatchesDataset(BaseDataset):
    """HPatches evaluation dataset.

    Follows the evaluation protocol from SuperPoint and other keypoint papers.
    Optionally ignores large scenes for fair comparison.
    """

    default_conf = {
        "data_dir": "/mnt/e/datasets/hpatches-sequences-release",
        "split": "test",
        "scene_type": "all",  # all, vantage, illumination
        "ignore_large_scenes": True,  # Ignore scenes in IGNORED_SCENES
        "max_scenes": None,  # Limit number of scenes for fast eval (None = all)
        "max_pairs_per_scene": None,  # Limit pairs per scene (None = all 5 pairs)
    }

    def _init(self, conf):
        data_dir = Path(conf.data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"HPatches dataset not found: {data_dir}")

        # Get all sequences
        all_sequences = [d.name for d in data_dir.iterdir() if d.is_dir()]

        # Filter by scene type
        if conf.scene_type == "vantage":
            sequences = [s for s in all_sequences if s.startswith("v_")]
        elif conf.scene_type == "illumination":
            sequences = [s for s in all_sequences if s.startswith("i_")]
        else:
            sequences = all_sequences

        # Filter out large scenes if configured
        if conf.ignore_large_scenes:
            sequences = [s for s in sequences if s not in IGNORED_SCENES]

        sequences = sorted(sequences)
        self.sequences = sequences
        self.data_dir = data_dir
        logger.info(f"Loaded {len(sequences)} HPatches sequences "
              f"({conf.scene_type}, ignore_large={conf.ignore_large_scenes})")

    def get_dataset(self, _):
        # HPatches only has test split, split parameter is intentionally unused
        return _HPatchesDataset(self.conf, self.sequences, self.data_dir)


class _HPatchesDataset(torch.utils.data.Dataset):
    """Internal HPatches dataset that generates all image pairs."""

    def __init__(self, conf, sequences, data_dir):
        self.conf = conf
        self.sequences = sequences
        self.data_dir = data_dir

        # Apply max_scenes limit if configured (for fast eval)
        max_scenes = conf.get("max_scenes")
        if max_scenes is not None and len(sequences) > max_scenes:
            # Select evenly distributed scenes for representative sampling
            indices = np.linspace(0, len(sequences) - 1, max_scenes, dtype=int)
            sequences = [sequences[i] for i in indices]
            logger.info(f"Limited eval to {max_scenes} scenes: {sequences}")

        # Build list of all image pairs
        self.pairs = []
        max_pairs = conf.get("max_pairs_per_scene")

        for seq in sequences:
            seq_dir = data_dir / seq
            is_illumination = seq.startswith("i_")
            # Find all image pairs (1 vs 2-6)
            pair_count = 0
            for i in range(2, 7):
                if max_pairs is not None and pair_count >= max_pairs:
                    break

                img1_path = seq_dir / "1.ppm"
                img2_path = seq_dir / f"{i}.ppm"
                H_path = seq_dir / f"H_1_{i}"

                if img1_path.exists() and img2_path.exists() and H_path.exists():
                    self.pairs.append((seq, img1_path, img2_path, H_path,
                                       is_illumination, i))
                    pair_count += 1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq, img1_path, img2_path, H_path, is_illumination, img_idx = self.pairs[idx]

        # Load images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        if img1 is None or img2 is None:
            # Return dummy data if loading fails
            img1 = np.zeros((480, 640, 3), dtype=np.uint8)
            img2 = np.zeros((480, 640, 3), dtype=np.uint8)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Load homography using robust reader
        try:
            H = read_homography(H_path)
        except Exception:
            H = np.eye(3, dtype=np.float32)

        # Store original image sizes before normalization
        orig_size = (img1.shape[1], img1.shape[0])  # (W, H)

        # Convert to tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img1 = (img1 - mean) / std
        img2 = (img2 - mean) / std

        return {
            "image0": {"image": img1},
            "image1": {"image": img2},
            "H_0to1": torch.from_numpy(H).float(),
            "seq_name": seq,
            "pair_idx": idx,
            "is_illumination": is_illumination,
            "img_idx": img_idx,
            "image_size": torch.tensor(orig_size),
        }
