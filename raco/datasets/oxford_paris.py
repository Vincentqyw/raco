"""Oxford/Paris dataset for RaCo training.

Incorporates patterns from glue-factory:
- sample_homography_corners: proper corner-based homography sampling
- ImagePreprocessor: unified image preprocessing with scale tracking
- Photometric augmentation via Albumentations
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from .base_dataset import BaseDataset


def create_center_patch(shape, patch_shape=None):
    """Create a centered patch within the image."""
    if patch_shape is None:
        patch_shape = shape
    width, height = shape
    pwidth, pheight = patch_shape
    left = int((width - pwidth) / 2)
    bottom = int((height - pheight) / 2)
    right = int((width + pwidth) / 2)
    top = int((height + pheight) / 2)
    return np.array([[left, bottom], [left, top], [right, top], [right, bottom]])


def check_convex(patch, min_convexity=0.05):
    """Check if polygon vertices form a convex shape."""
    for i in range(patch.shape[0]):
        x1, y1 = patch[(i - 1) % patch.shape[0]]
        x2, y2 = patch[i]
        x3, y3 = patch[(i + 1) % patch.shape[0]]
        if (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1) > -min_convexity:
            return False
    return True


def sample_homography_corners(
    shape,
    patch_shape=None,
    difficulty=0.8,
    translation=1.0,
    max_angle=60,
    n_angles=10,
    min_convexity=0.05,
    rng=None,
):
    """Sample random homography by perturbing image corners.

    Based on glue-factory's sample_homography_corners.

    Args:
        shape: (width, height) of source image
        patch_shape: (width, height) of output patch, None for same as shape
        difficulty: 0-1 scale of how difficult the homography should be
        translation: amount of translation (relative to image size)
        max_angle: max rotation angle in degrees
        n_angles: number of rotation angles to try
        min_convexity: minimum convexity for valid patch
        rng: numpy random state

    Returns:
        H: (3, 3) homography matrix
        corners_src: (4, 2) source corners
        corners_dst: (4, 2) destination corners
    """
    if rng is None:
        rng = np.random

    max_angle = max_angle / 180.0 * math.pi
    width, height = shape

    if patch_shape is None:
        patch_shape = shape

    pwidth, pheight = width * (1 - difficulty), height * (1 - difficulty)
    min_pts1 = create_center_patch(shape, (pwidth, pheight))
    full = create_center_patch(shape)
    pts2 = create_center_patch(patch_shape)
    scale = min_pts1 - full

    # Sample valid convex quadrilateral
    found_valid = False
    while not found_valid:
        offsets = rng.uniform(0.0, 1.0, size=(4, 2)) * scale
        pts1 = full + offsets
        found_valid = check_convex(pts1 / np.array(shape), min_convexity)

    # Re-center
    pts1 = pts1 - np.mean(pts1, axis=0, keepdims=True)
    pts1 = pts1 + np.mean(min_pts1, axis=0, keepdims=True)

    # Rotation
    if n_angles > 0 and difficulty > 0:
        angles = np.linspace(-max_angle * difficulty, max_angle * difficulty, n_angles)
        rng.shuffle(angles)
        angles = np.concatenate([[0.0], angles], axis=0)

        center = np.mean(pts1, axis=0, keepdims=True)
        rot_mat = np.reshape(
            np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)],
                axis=1,
            ),
            [-1, 2, 2],
        )
        rotated = (
            np.matmul(
                np.tile(np.expand_dims(pts1 - center, axis=0), [n_angles + 1, 1, 1]),
                rot_mat,
            )
            + center
        )

        for idx in range(1, n_angles):
            warped_points = rotated[idx] / np.array(shape)
            if np.all((warped_points >= 0.0) & (warped_points < 1.0)):
                pts1 = rotated[idx]
                break

    # Translation
    if translation > 0:
        min_trans = -np.min(pts1, axis=0)
        max_trans = shape - np.max(pts1, axis=0)
        trans = rng.uniform(min_trans, max_trans)[None]
        pts1 += trans * translation * difficulty

    # Compute homography from 4 point correspondences
    H = compute_homography(pts1, pts2)
    return H.astype(np.float32), full.astype(np.float32), pts1.astype(np.float32)


def compute_homography(pts1, pts2, shape=None):
    """Compute homography from 4 point correspondences.

    Args:
        pts1: (4, 2) source points in pixels
        pts2: (4, 2) destination points in pixels
        shape: deprecated, kept for compatibility

    Returns:
        H: (3, 3) homography matrix
    """
    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(
        np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0)
    )
    homography = np.transpose(np.linalg.solve(a_mat, p_mat))
    # Flatten to 3x3
    homography = np.concatenate([homography.flatten(), [1.0]])
    return homography.reshape(3, 3)


def warp_points(points, H, inverse=True):
    """Warp points using homography.

    Args:
        points: (N, 2) array of points
        H: (3, 3) homography matrix
        inverse: whether to use H or H^{-1}

    Returns:
        warped: (N, 2) array of warped points
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    H_mat = np.linalg.inv(H) if inverse else H
    warped = (H_mat @ points_h.T).T
    return warped[:, :2] / warped[:, 2:3]


class ImagePreprocessor:
    """Image preprocessing with resizing and scale tracking.

    Similar to glue-factory's ImagePreprocessor but simpler.
    """

    def __init__(self, resize=None, side="short"):
        self.resize = resize
        self.side = side

    def __call__(self, img):
        """Preprocess image and return with metadata.

        Args:
            img: (H, W, C) numpy array

        Returns:
            dict with 'image', 'image_size', 'transform', 'original_image_size'
        """
        h, w = img.shape[:2]
        size = (h, w)
        scale = np.array([1.0, 1.0])

        if self.resize is not None:
            # For training, resize both sides to the target size
            # This ensures all images in a batch have the same size
            if isinstance(self.resize, (list, tuple)) and len(self.resize) == 2:
                new_h, new_w = self.resize
            elif self.side == "fixed":
                # resize is (H, W) tuple
                new_h, new_w = self.resize
            elif self.side == "short":
                if h < w:
                    new_h = self.resize
                    new_w = int(w * self.resize / h)
                else:
                    new_w = self.resize
                    new_h = int(h * self.resize / w)
            elif self.side == "long":
                if h > w:
                    new_h = self.resize
                    new_w = int(w * self.resize / h)
                else:
                    new_w = self.resize
                    new_h = int(h * self.resize / w)
            else:  # both
                new_w = new_h = self.resize

            size = (new_h, new_w)
            scale = np.array([new_w / w, new_h / h])
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        transform = np.diag([scale[0], scale[1], 1.0])

        return {
            "image": img,
            "image_size": np.array([size[1], size[0]]),  # (W, H)
            "transform": transform,
            "original_image_size": np.array([w, h]),  # (W, H)
            "scale": scale,
        }


def photometric_augmentation(image, aug_params=None):
    """Apply photometric augmentation.

    Args:
        image: (H, W, C) numpy array, uint8 or float32
        aug_params: dict with augmentation parameters

    Returns:
        augmented image
    """
    if aug_params is None:
        aug_params = {}

    p = aug_params.get("p", 0.5)

    # Convert to float if needed
    was_uint8 = image.dtype == np.uint8
    if was_uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    # Brightness
    if np.random.rand() < p:
        factor = 1.0 + np.random.uniform(-0.3, 0.3)
        img = np.clip(img * factor, 0, 1)

    # Contrast
    if np.random.rand() < p:
        factor = 1.0 + np.random.uniform(-0.3, 0.3)
        img = np.clip((img - 0.5) * factor + 0.5, 0, 1)

    # Gamma / RandomAdditiveShade-like effect
    if np.random.rand() < p * 0.5:
        gamma = np.random.uniform(0.7, 1.5)
        img = np.clip(img ** gamma, 0, 1)

    # Noise
    if np.random.rand() < p * 0.5:
        noise = np.random.randn(*img.shape) * 0.05
        img = np.clip(img + noise, 0, 1)

    # Blur
    if np.random.rand() < p * 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Convert back
    if was_uint8:
        return (img * 255).astype(np.uint8)
    return img


class OxfordParisDataset(BaseDataset):
    """Oxford/Paris dataset for RaCo training with improved homography sampling."""

    default_conf = {
        "data_root": "/mnt/e/datasets/raco",
        "datasets": ["roxford5k", "rparis6k"],
        "image_size": [640, 640],
        "train_size": 5000,
        "val_size": 200,
        "shuffle_seed": 0,
        "grayscale": False,
        "homography": {
            "difficulty": 0.8,  # 0-1 difficulty scale
            "translation": 1.0,
            "max_angle": 60,  # degrees
            "n_angles": 10,
            "min_convexity": 0.05,
            "patch_shape": None,  # None for same as image_size
        },
        "photometric": {
            "p": 0.5,
        },
    }

    def _init(self, conf):
        data_root = Path(conf.data_root)

        all_images = []
        for dataset_name in conf.datasets:
            image_dir = data_root / dataset_name / "jpg"
            if image_dir.exists():
                images = sorted(image_dir.glob("*.jpg"))
                if len(images) == 0:
                    images = sorted(image_dir.glob("*.png"))
                all_images.extend([(dataset_name, i.name) for i in images])
                logger.info(f"Loaded {len(images)} images from {dataset_name}")

        if len(all_images) == 0:
            raise ValueError(f"No images found in {conf.datasets}")

        # Shuffle and split
        if conf.shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(all_images)

        train_images = all_images[:conf.train_size]
        val_images = all_images[conf.train_size:conf.train_size + conf.val_size]

        self.images = {"train": train_images, "val": val_images}
        self.data_root = data_root

    def get_dataset(self, split):
        return _Dataset(self.conf, self.images[split], split, self.data_root)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_list, split, data_root):
        self.conf = conf
        self.image_list = image_list
        self.split = split
        self.data_root = data_root

        # Setup image preprocessor - use fixed size for batching
        if conf.image_size:
            # Use fixed (H, W) size to ensure all images same size for batching
            target_size = (conf.image_size[1], conf.image_size[0])  # (H, W)
        else:
            target_size = None
        self.preprocessor = ImagePreprocessor(
            resize=target_size,
            side="fixed"
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        dataset_name, img_name = self.image_list[idx]
        img_path = self.data_root / dataset_name / "jpg" / img_name

        # Load image (similar to glue-factory)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            # Return dummy data if loading fails
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess (resize if needed)
        data0 = self.preprocessor(image)
        img_shape = data0["image"].shape[:2]  # (H, W)

        # Generate homography using corner-based sampling
        h_conf = self.conf.homography
        patch_shape = h_conf.get("patch_shape")
        if patch_shape is None:
            patch_shape = (data0["image_size"][0], data0["image_size"][1])  # (W, H)

        H, corners_src, corners_dst = sample_homography_corners(
            shape=(data0["image_size"][0], data0["image_size"][1]),  # (W, H)
            patch_shape=patch_shape,
            difficulty=h_conf.get("difficulty", 0.8),
            translation=h_conf.get("translation", 1.0),
            max_angle=h_conf.get("max_angle", 60),
            n_angles=h_conf.get("n_angles", 10),
            min_convexity=h_conf.get("min_convexity", 0.05),
        )

        # Apply homography to create warped view
        h, w = img_shape
        image_warped = cv2.warpPerspective(
            data0["image"],
            H,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

        # Photometric augmentation (different for each view)
        img_aug = photometric_augmentation(
            data0["image"],
            self.conf.photometric
        )
        img_warped_aug = photometric_augmentation(
            image_warped,
            self.conf.photometric
        )

        # Convert to tensor
        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
        img_warped_tensor = torch.from_numpy(img_warped_aug).permute(2, 0, 1).float() / 255.0

        # Normalize (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_warped_tensor = (img_warped_tensor - mean) / std

        # Compute H in relative coordinates (accounting for resize)
        # H maps from original image0 to warped image1
        # We need to account for the preprocessing transforms
        T = data0["transform"]  # scale transform
        H_tensor = torch.from_numpy(H).float()

        return {
            "view0": {
                "image": img_tensor,
                "original_image_size": data0["original_image_size"],
            },
            "view1": {
                "image": img_warped_tensor,
                "original_image_size": data0["original_image_size"],
            },
            "H_0to1": H_tensor,
            "name": f"{dataset_name}/{img_name}",
            "image_size": torch.from_numpy(data0["image_size"]).float(),
        }


# Register dataset
__dataset__ = OxfordParisDataset
