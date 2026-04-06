"""RaCo (Ranking and Covariance) Feature Extractor."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Optional
from loguru import logger
from ..base_model import BaseModel
from ...utils.utils import ImagePreprocessor


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=kernel_size // 2, bias=False,
    )


class InputPadder:
    """Pads images such that dimensions are divisible by N."""

    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2, pad_wd - pad_wd // 2,
            pad_ht // 2, pad_ht - pad_ht // 2,
        ]

    def pad(self, x):
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gate = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))
        x = self.gate(self.bn2(self.conv2(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate = nn.SELU(inplace=True)
        self.match_dims = nn.Conv2d(inplanes, planes, 1, 1)

    def forward(self, x):
        identity = self.match_dims(x)
        out = self.gate(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.gate(out)


def _covariance_matrix_from_cholesky(cholesky_vec):
    """Convert Cholesky factors (L11, L21, L22) to covariance matrix."""
    L11, L21, L22 = torch.unbind(cholesky_vec, dim=-1)
    zeros = torch.zeros_like(L11)
    L = torch.stack([
        torch.stack([L11, zeros], dim=-1),
        torch.stack([L21, L22], dim=-1)
    ], dim=-2)

    if cholesky_vec.dim() > 1:
        shape = cholesky_vec.shape[:-1]
        L_flat = L.reshape(-1, 2, 2)
        cov_flat = torch.bmm(L_flat, L_flat.transpose(-1, -2))
        return cov_flat.reshape(shape + (2, 2))
    return torch.matmul(L, L.transpose(-1, -2))


def _extract_patches(x, inds, patch_size):
    """Extract patches at indices."""
    B, H, W = x.shape
    N = inds.shape[1]
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded = unfolder(x[:, None])
    patches = torch.gather(unfolded, dim=2, index=inds[:, None, :].expand(B, patch_size**2, N))
    return patches


def _compute_subpixel_offsets(raw_logits, inds, nms_radius, temp=0.5):
    """Compute subpixel offsets."""
    B = raw_logits.shape[0]
    device = raw_logits.device

    offset_range = torch.linspace(-(nms_radius - 1) / 2, (nms_radius - 1) / 2, nms_radius, device=device)
    offset_grid = torch.meshgrid(offset_range, offset_range, indexing="ij")
    offsets = torch.stack((offset_grid[1], offset_grid[0]), dim=-1).reshape(nms_radius**2, 2)
    offsets = offsets.unsqueeze(0).expand(B, -1, -1)

    patch_scores = _extract_patches(raw_logits.squeeze(1), inds, nms_radius)
    patch_probs = (patch_scores / temp).softmax(dim=1)
    return torch.einsum("bkn,bkd->bnd", patch_probs, offsets)


def _sample_at_positions(feature_map, positions, H, W, use_subpixel):
    """Sample feature map at positions."""
    B, C = feature_map.shape[:2]

    if use_subpixel:
        grid = torch.stack([
            2.0 * positions[..., 0] / (W - 1) - 1.0,
            2.0 * positions[..., 1] / (H - 1) - 1.0,
        ], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True).squeeze(-1)
    else:
        idxs = torch.round(positions[..., 1]).long() * W + torch.round(positions[..., 0]).long()
        idxs = idxs.clamp(0, W * H - 1)
        sampled = feature_map.reshape(B, C, -1).gather(2, idxs.unsqueeze(1).expand(B, C, -1))

    sampled = sampled.permute(0, 2, 1)
    return sampled.squeeze(-1) if C == 1 else sampled


class RaCo(BaseModel):
    """RaCo: Ranking and Covariance for Learned Keypoints."""

    default_conf = {
        "max_num_keypoints": 2048,
        "nms_radius": 3,
        "subpixel_sampling": True,
        "subpixel_temp": 0.5,
        "ranker": True,
        "covariance_estimator": True,
        "sort_by_ranker": False,
        "remove_borders": True,  # Remove keypoints near borders (following DaD)
        "border_size": 4,  # Border size in pixels to remove
        "detector": {
            "d_max": 1.2,
            "rho_pos": 1.0,
            "rho_neg_max": 1e-2,
        },
        "ranker": {
            "lambda_ranker": 1.0,
        },
        "weights": None,
    }
    preprocess_conf = {
        "resize": None,
    }
    required_data_keys = ["image0", "image1"]
    strict_conf = False

    def _init(self, conf):
        # Normalization
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    
        # Architecture
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)
        self.gate = nn.SELU(inplace=True)

        c1, c2, c3, c4 = 16, 32, 64, 128
        self.block1 = ConvBlock(3, c1)
        self.block2 = ResBlock(c1, c2)
        self.block3 = ResBlock(c2, c3)
        self.block4 = ResBlock(c3, c4)

        self.conv1 = conv1x1(c1, c4 // 4)
        self.conv2 = conv3x3(c2, c4 // 4)
        self.conv3 = conv3x3(c3, c4 // 4)
        self.conv4 = conv3x3(c4, c4 // 4)

        # Score head
        self.score_head = nn.Sequential(
            conv1x1(c4, 8), nn.SELU(inplace=True),
            conv3x3(8, 4), nn.SELU(inplace=True),
            conv3x3(4, 4), nn.SELU(inplace=True),
            conv3x3(4, 1),
        )

        # Ranker head
        if conf.ranker:
            ranker_dim = 12
            ranker_layers = [ResBlock(3, ranker_dim)]
            ranker_layers += [ResBlock(ranker_dim, ranker_dim) for _ in range(8)]
            ranker_layers.append(nn.Conv2d(ranker_dim, 1, 5, padding=2, bias=True, padding_mode="reflect"))
            self.ranker_head = nn.Sequential(*ranker_layers)

        # Covariance head
        if conf.covariance_estimator:
            cov_modules = []
            in_ch = c4
            for out_ch in [64, 32, 32]:
                cov_modules.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False, padding_mode="reflect"))
                cov_modules.append(nn.LeakyReLU(inplace=True))
                in_ch = out_ch
            cov_modules.append(nn.Conv2d(32, 3, 1, bias=True, padding_mode="reflect"))
            self.covariance_estimator_head = nn.Sequential(*cov_modules)
            self.var_activation = nn.Softplus()

            # Initialize the last layer to output reasonable covariance values
            # Default init gives small values, so we initialize bias to give larger initial covariances
            # This prevents negative NLL in early training
            with torch.no_grad():
                # Set bias to give initial log-variance around 0 -> variance around 1
                # Output is [var_x, cov_xy, var_y]
                # We want softplus(bias_x) ≈ 1.5, softplus(bias_y) ≈ 1.5
                # softplus(x) ≈ x for x >> 0, so bias ≈ 1.5 for vars, 0 for cov
                self.covariance_estimator_head[-1].bias.data[:] = torch.tensor([0.5, 0.0, 0.5])

        if self.conf.weights is not None:
            # Load pretrained weights from URL or local path
            if isinstance(self.conf.weights, str) and self.conf.weights.startswith(
                ("http://", "https://")
            ):
                state_dict = torch.hub.load_state_dict_from_url(
                    self.conf.weights,
                    map_location="cpu",
                    progress=True,
                    weights_only=True,
                )
            else:
                state_dict = torch.load(
                    self.conf.weights, map_location="cpu", weights_only=True
                )

            self.load_state_dict(state_dict, strict=False)
            logger.info(f"[RaCo] Loaded weights from {self.conf.weights}")
        else:
            logger.warning(f"[RaCo] weight is None")


    def forward_dual(self, data):
        """Forward pass returning predictions."""
        pred = {}

        for i, view_key in enumerate(["image0", "image1"]):
            view = data[view_key]  # key: image
            # Extract features
            features = self._forward(view)
            pred[view_key] = features

            # Rename for convenience
            pred[f"keypoints_{i}"] = features["keypoints"]
            pred[f"keypoint_scores_{i}"] = features["keypoint_scores"]
            if "ranker_scores" in features:
                pred[f"ranker_scores_{i}"] = features["ranker_scores"]
            if "covariances" in features:
                pred[f"covariances_{i}"] = features["covariances"]
            if "raw_scores" in features:
                pred[f"raw_scores_{i}"] = features["raw_scores"]
            if "prob_map" in features:
                pred[f"prob_map_{i}"] = features["prob_map"]
            if "ranker_map" in features:
                pred[f"ranker_map_{i}"] = features["ranker_map"]
            if "covariances_map" in features:
                pred[f"covariances_map_{i}"] = features["covariances_map"]

        return pred


    def _forward(self, data: dict) -> dict:
        # Preprocess image
        image = data["image"]
        return self.forward_single(image)


    def forward_single(self, image):
        """Extract features from a single image."""
        # Preprocess image
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # Convert to 3-channel greyscale
        image = self.normalizer(image)  # (x-mean) / std

        # Pad to divisible by 32
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        x = padder.pad(image)

        # Encoder
        x1 = self.block1(x)
        x2 = self.pool2(x1)
        x2 = self.block2(x2)
        x3 = self.pool4(x2)
        x3 = self.block3(x3)
        x4 = self.pool4(x3)
        x4 = self.block4(x4)

        # Aggregation
        x1_up = self.gate(self.conv1(x1))
        x2_up = F.interpolate(self.gate(self.conv2(x2)), scale_factor=2, mode="bilinear", align_corners=True)
        x3_up = F.interpolate(self.gate(self.conv3(x3)), scale_factor=8, mode="bilinear", align_corners=True)
        x4_up = F.interpolate(self.gate(self.conv4(x4)), scale_factor=32, mode="bilinear", align_corners=True)
        x_fused = torch.cat([x1_up, x2_up, x3_up, x4_up], dim=1)

        # Score head - output logits (not probabilities)
        raw_scores = self.score_head(x_fused)
        raw_scores = padder.unpad(raw_scores)

        # Probability map (global softmax for visualization and sampling)
        B, _, H, W = raw_scores.shape
        prob_map = F.softmax(raw_scores.flatten(1), dim=1).reshape(raw_scores.size())

        # Debug: log probability distribution statistics during training
        if self.training and B > 0:
            # Check if prob_map is too flat or too sharp
            prob_max = prob_map.max()
            prob_min = prob_map.min()
            prob_entropy = -(prob_map * torch.log(prob_map + 1e-8)).sum() / B
            # Store for logging (can be accessed externally)
            self._debug_prob_stats = {
                "prob_max": prob_max.item(),
                "prob_min": prob_min.item(),
                "prob_entropy": prob_entropy.item(),
                "prob_mean": prob_map.mean().item(),
                "raw_scores_std": raw_scores.std().item(),
            }

        kpts = self._sample_keypoints(prob_map, raw_scores)

        probs = _sample_at_positions(prob_map, kpts, H, W, self.conf.subpixel_sampling)  # (B, N)

        result = {
            "raw_scores": raw_scores,  # Logits for loss computation
            "prob_map": prob_map,  # Softmax probabilities for visualization
            "keypoints": kpts,
            "keypoint_scores": probs,
        }

        # Ranker
        if self.conf.ranker:
            ranker_feat = self.ranker_head(x)
            ranker_feat = padder.unpad(ranker_feat)
            ranker_scores = _sample_at_positions(
                ranker_feat, kpts, H, W, self.conf.subpixel_sampling
            )
            result["ranker_scores"] = ranker_scores
            if not self.training:
                result["ranker_map"] = ranker_feat  # B x 1 x H x W

        # Covariance
        if self.conf.covariance_estimator:
            cov_feat = self.covariance_estimator_head(x_fused)
            cov_feat = padder.unpad(cov_feat)
            cov_feat = torch.stack([
                self.var_activation(cov_feat[:, 0]),
                cov_feat[:, 1],
                self.var_activation(cov_feat[:, 2]),
            ], dim=1)
            cov_values = _sample_at_positions(
                cov_feat, kpts, H, W, self.conf.subpixel_sampling
            )
            result["covariances"] = _covariance_matrix_from_cholesky(cov_values)  # (B, N, 2, 2)
            if not self.training:
                result["covariances_map"] = cov_feat  # B x 3 x H x W

        # Sort by ranker if requested
        if self.conf.sort_by_ranker and "ranker_scores" in result:
            sort_idx = torch.argsort(result["ranker_scores"], dim=1, descending=True)
            result["keypoints"] = torch.gather(
                result["keypoints"], 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2)
            )
            result["keypoint_scores"] = torch.gather(result["keypoint_scores"], 1, sort_idx)
            result["ranker_scores"] = torch.gather(result["ranker_scores"], 1, sort_idx)
            if "covariances" in result:
                result["covariances"] = torch.gather(
                    result["covariances"], 1, sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
                )

        return result


    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing.

        Args:
            img: Input image tensor of shape (C, H, W) or (B, C, H, W)
            **conf: Additional preprocessing configuration (e.g., resize settings)

        Returns:
            Dictionary containing extracted features with scaled coordinates
        """
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward_single(img)
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = feats["keypoints"] / scales[None]

        # Scale covariances if present
        if "covariances" in feats:
            scales_mat = torch.diag(scales).to(img)
            feats["covariances"] = (
                scales_mat[None]
                @ feats["covariances"]
                @ scales_mat[None].transpose(-1, -2)
            )
        return feats

    def _sample_keypoints(self, prob_map, raw_scores: Optional[torch.Tensor] = None, sample_topk=True):
        """Sample keypoints from probability map with NMS."""
        # B = prob_map.shape[0]
        B, C, H, W = prob_map.size()
        num_kpts = min(self.conf.max_num_keypoints, H * W)
        nms_radius = self.conf.nms_radius

        # Remove borders (following DaD)
        if self.conf.remove_borders:
            border = self.conf.border_size
            prob_map_masked = prob_map.clone()
            prob_map_masked[..., :border, :] = 0  # Top border
            prob_map_masked[..., -border:, :] = 0  # Bottom border
            prob_map_masked[..., :, :border] = 0  # Left border
            prob_map_masked[..., :, -border:] = 0  # Right border
        else:
            prob_map_masked = prob_map

        # NMS: keep only local maxima
        # Use max_pool2d to find local maxima in each nms_radius x nms_radius window
        # Use kernel_size=nms_radius (3) not 2*radius+1 to match DAD's nms_size=3
        max_pooled = F.max_pool2d(
            prob_map_masked,
            kernel_size=nms_radius,
            stride=1,
            padding=nms_radius // 2
        )
        # A pixel is a local maximum if it's equal to the max in its neighborhood
        is_local_max = (prob_map_masked == max_pooled)
        prob_nms = prob_map_masked * is_local_max.float()

        # Top-k: select topk keypoints from NMS-suppressed map
        prob_flat = prob_nms.reshape(B, H * W)

        if sample_topk:
            hw_inds = torch.topk(prob_flat, k=num_kpts, dim=1).indices
        else:
            hw_inds = torch.multinomial(prob_flat, num_samples=num_kpts, replacement=False)

        h_inds = hw_inds // W
        w_inds = hw_inds % W
        kpts = torch.stack([w_inds.float(), h_inds.float()], dim=-1)   # (B, num_kpts, 2)

        # Subpixel refinement
        if self.conf.subpixel_sampling and raw_scores is not None:
            offsets = _compute_subpixel_offsets(raw_scores, hw_inds, nms_radius, self.conf.subpixel_temp)
            kpts = kpts + offsets

        return kpts + 0.5

    def loss(self, pred, data):
        """Compute losses - to be implemented with proper loss functions."""
        return {"total": torch.tensor(0.0, device=pred["keypoints_0"].device)}

    def metrics(self, pred, data):
        """Compute metrics."""
        return {}


# Register model
__model__ = RaCo
