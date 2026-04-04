"""RaCo (Ranking and Covariance) Feature Extractor."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


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
        "detector": {
            "d_max": 1.2,
            "rho_pos": 1.0,
            "rho_neg_max": 1e-2,
        },
        "ranker": {
            "lambda_ranker": 1.0,
        },
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False

    def _init(self, conf):
        # Normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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
        if conf.use_ranker:
            ranker_dim = 12
            ranker_layers = [ResBlock(3, ranker_dim)]
            ranker_layers += [ResBlock(ranker_dim, ranker_dim) for _ in range(8)]
            ranker_layers.append(nn.Conv2d(ranker_dim, 1, 5, padding=2, bias=True, padding_mode="reflect"))
            self.ranker_head = nn.Sequential(*ranker_layers)

        # Covariance head
        if conf.use_covariance:
            cov_modules = []
            in_ch = c4
            for out_ch in [64, 32, 32]:
                cov_modules.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False, padding_mode="reflect"))
                cov_modules.append(nn.LeakyReLU(inplace=True))
                in_ch = out_ch
            cov_modules.append(nn.Conv2d(32, 3, 1, bias=True, padding_mode="reflect"))
            self.covariance_head = nn.Sequential(*cov_modules)
            self.cov_activation = nn.Softplus()

    def _normalize(self, image):
        return (image - self.mean) / self.std

    def _forward(self, data):
        """Forward pass returning predictions."""
        pred = {}

        for i, view_key in enumerate(["view0", "view1"]):
            view = data[view_key]
            image = view["image"]
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)

            # Extract features
            features = self._extract_features(image)
            pred[view_key] = features

            # Rename for convenience
            pred[f"keypoints_{i}"] = features["keypoints"]
            pred[f"keypoint_scores_{i}"] = features["keypoint_scores"]
            if "ranker_scores" in features:
                pred[f"ranker_scores_{i}"] = features["ranker_scores"]
            if "covariances" in features:
                pred[f"covariances_{i}"] = features["covariances"]

        return pred

    def _extract_features(self, image):
        """Extract features from a single image."""
        image = self._normalize(image)

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

        # Score head
        raw_scores = self.score_head(x_fused)
        raw_scores = padder.unpad(raw_scores)

        # Probability map (global softmax)
        B, _, H, W = raw_scores.shape
        prob_map = F.softmax(raw_scores.flatten(1), dim=1).reshape(raw_scores.shape)

        # Sample keypoints
        kpts, scores_dict = self._sample_keypoints(prob_map, raw_scores, H, W)

        result = {
            "raw_scores": raw_scores,
            "prob_map": prob_map,
            "keypoints": kpts,
            "keypoint_scores": scores_dict["detection_scores"],
        }

        # Ranker
        if self.conf.use_ranker:
            ranker_feat = self.ranker_head(x)
            ranker_feat = padder.unpad(ranker_feat)
            ranker_scores = _sample_at_positions(
                ranker_feat, kpts, H, W, self.conf.subpixel_sampling
            )
            result["ranker_scores"] = ranker_scores

        # Covariance
        if self.conf.use_covariance:
            cov_feat = self.covariance_head(x_fused)
            cov_feat = padder.unpad(cov_feat)
            cov_feat = torch.stack([
                self.cov_activation(cov_feat[:, 0]),
                cov_feat[:, 1],
                self.cov_activation(cov_feat[:, 2]),
            ], dim=1)
            cov_values = _sample_at_positions(
                cov_feat, kpts, H, W, self.conf.subpixel_sampling
            )
            result["covariances"] = _covariance_matrix_from_cholesky(cov_values)

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

    def _sample_keypoints(self, prob_map, raw_scores, H, W):
        """Sample keypoints from probability map."""
        B = prob_map.shape[0]
        num_kpts = min(self.conf.max_num_keypoints, H * W)
        nms_radius = self.conf.nms_radius

        # NMS
        max_pooled = F.max_pool2d(prob_map, nms_radius, stride=1, padding=nms_radius // 2)
        prob_nms = prob_map * (prob_map == max_pooled)

        # Top-k
        prob_flat = prob_nms.reshape(B, H * W)
        topk = torch.topk(prob_flat, k=num_kpts, dim=1)

        hw_inds = topk.indices
        h_inds = hw_inds // W
        w_inds = hw_inds % W
        kpts = torch.stack([w_inds.float(), h_inds.float()], dim=-1)

        # Subpixel refinement
        if self.conf.subpixel_sampling:
            offsets = _compute_subpixel_offsets(raw_scores, hw_inds, nms_radius, self.conf.subpixel_temp)
            kpts = kpts + offsets

        # Get scores at keypoints
        scores = topk.values

        return kpts + 0.5, {"detection_scores": scores}

    def loss(self, pred, data):
        """Compute losses - to be implemented with proper loss functions."""
        return {"total": torch.tensor(0.0, device=pred["keypoints_0"].device)}

    def metrics(self, pred, data):
        """Compute metrics."""
        return {}


# Register model
__model__ = RaCo
