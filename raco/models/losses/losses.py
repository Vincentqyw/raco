"""Loss functions for RaCo trainer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Tuple, Dict, Optional
from .soft_rank import soft_rank
from ..reward_functions import ConstantReward


class DetectorLoss(nn.Module):
    """
    Keypoint detector loss using policy gradient (Eq. 3 in paper).
    L_detector = -sum(rho' * log(pi))

    Reference: "Learning Feature Descriptors using Deep Neural Networks"
    """

    def __init__(
        self,
        reward_fn: Optional[nn.Module] = None,
        d_max: float = 1.2,
        rho_pos: float = 1.0,
        rho_neg_max: float = 1e-2,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.reward_fn = reward_fn or ConstantReward(
            d_max=d_max, rho_pos=rho_pos, rho_neg_max=rho_neg_max, epsilon=epsilon
        )
        self.epsilon = epsilon

    def compute_reward(
        self, distances: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute reward using the configured reward function."""
        return self.reward_fn(distances, valid_mask)

    def normalize_reward(
        self, rewards: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Normalize reward following DaD paper:
        rho' = rho / (E[rho] + epsilon)

        Also handles early exit when no valid keypoints exist.

        Args:
            rewards: (B, N) tensor of rewards
            valid_mask: (B, N) boolean tensor indicating valid keypoints

        Returns:
            (normalized_rewards, should_exit) tuple:
                - normalized_rewards: (B, N) tensor
                - should_exit: bool indicating if no valid keypoints (loss should be zero)
        """
        if valid_mask is not None:
            # Compute mean over valid keypoints only
            valid_rewards = rewards * valid_mask.float()
            reward_sum = valid_rewards.sum(dim=1, keepdim=True)
            valid_count = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            reward_mean = reward_sum / valid_count

            # Check if any valid keypoints exist
            if valid_count.sum().item() == 0:
                return rewards, True

        else:
            reward_mean = rewards.mean(dim=1, keepdim=True)

        # Add epsilon for numerical stability
        normalized_rewards = rewards / (reward_mean + self.epsilon)
        return normalized_rewards, False

    def forward(
        self,
        prob_map_flat: torch.Tensor,
        distances: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute detector loss."""
        rewards = self.compute_reward(distances, valid_mask)
        normalized_rewards, should_exit = self.normalize_reward(rewards, valid_mask)

        if should_exit:
            return torch.tensor(0.0, device=prob_map_flat.device, requires_grad=True)

        # Clamp prob_map to avoid log(0)
        prob_map_clamped = prob_map_flat.clamp(min=self.epsilon, max=1.0)
        log_probs = torch.log(prob_map_clamped)
        loss_per_loc = -normalized_rewards * log_probs

        if valid_mask is not None:
            loss_per_loc = loss_per_loc * valid_mask
            num_valid = valid_mask.sum().clamp(min=1)
            loss = loss_per_loc.sum() / num_valid
        else:
            loss = loss_per_loc.mean()

        return loss

    def forward_sparse(
        self,
        prob_sparse: torch.Tensor,
        reprojection_errors: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute detector loss on sparse keypoints (following DaD).
        This is more stable than dense loss over all pixels.

        Args:
            prob_sparse: (B, N) probability distribution over N keypoints (from softmax)
            reprojection_errors: (B, N, 2, 1) or (B, N, 2) reprojection errors (dx, dy) for each keypoint
            valid_mask: (B, N) boolean mask for valid keypoints
        """
        # Compute Euclidean distance from reprojection errors, distances shape: (B, N)
        assert (
            reprojection_errors.dim() == 2
        ), f"Unexpected reprojection_errors dim: {reprojection_errors.dim()}"
        distances = reprojection_errors
        rewards = self.compute_reward(distances, valid_mask)

        normalized_rewards, should_exit = self.normalize_reward(rewards, valid_mask)
        if should_exit:
            logger.warning("No valid keypoints, return zero loss")
            return torch.tensor(0.0, device=prob_sparse.device, requires_grad=True), {
                "rewards": rewards,
                "distances": distances,
            }

        # Compute policy gradient loss
        log_probs = torch.log(prob_sparse.clamp(min=self.epsilon))
        loss = -(normalized_rewards * log_probs)

        if valid_mask is not None:
            loss = loss * valid_mask.float()
            loss = loss.sum() / valid_mask.float().sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss, {
            "rewards": rewards,
            "distances": distances,
        }

    def set_step(self, step: int):
        """Update training step for dynamic rho_neg."""
        if isinstance(self.reward_fn, ConstantReward):
            self.reward_fn.set_step(step)


class RankingLoss(nn.Module):
    """Keypoint ranking loss (Eq. 4-5 in paper)."""

    def __init__(
        self,
        lambda_ranker: float = 1.0,
        regularization_strength: float = 1.0,
    ):
        super().__init__()
        self.lambda_ranker = lambda_ranker
        self.regularization_strength = regularization_strength

        # Use internal soft_rank implementation (no external dependencies)
        self.soft_rank = soft_rank
        self.has_soft_sort = True

    def _soft_rank_approximation(
        self, scores: torch.Tensor, tau: float = 0.1
    ) -> torch.Tensor:
        """Differentiable soft rank approximation using softmax."""
        B, N = scores.shape
        scores_normalized = scores - scores.mean(dim=-1, keepdim=True)

        scores_i = scores_normalized.unsqueeze(2)
        scores_j = scores_normalized.unsqueeze(1)
        diff = scores_j - scores_i

        soft_comparison = torch.sigmoid(diff / tau)
        soft_rank_0indexed = soft_comparison.sum(dim=2)

        return soft_rank_0indexed + 1

    def forward(
        self,
        ranker_scores_a: torch.Tensor,
        ranker_scores_b: torch.Tensor,
        mutual_mask_a: torch.Tensor,
        nearest_idx_a: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute ranking loss (Eq. 4-5 in paper).

        Args:
            ranker_scores_a: (B, N) ranking scores for keypoints in view A
            ranker_scores_b: (B, N) ranking scores for keypoints in view B
            mutual_mask_a: (B, N) boolean mask indicating matched keypoints in A
            nearest_idx_a: (B, N) indices of corresponding matched keypoints in B

        Returns:
            loss: Total ranking loss
            dict: Dictionary with individual loss components
        """
        B, N = ranker_scores_a.shape
        device = ranker_scores_a.device

        # Compute soft ranks
        soft_ranks_a = self.soft_rank(
            ranker_scores_a,
            direction="DESCENDING",
            regularization_strength=self.regularization_strength,
        )
        soft_ranks_b = self.soft_rank(
            ranker_scores_b,
            direction="DESCENDING",
            regularization_strength=self.regularization_strength,
        )

        # Normalize ranks to [0, 1] for numerical stability
        # DESCENDING mode returns values in [-N, -1] where -1 is highest rank (best)
        # Convert to [0, 1] where -1 → 0 (best) and -N → 1 (worst)
        soft_ranks_a_norm = (-soft_ranks_a - 1) / (N - 1 + 1e-8)  # (B, N)
        soft_ranks_b_norm = (-soft_ranks_b - 1) / (N - 1 + 1e-8)  # (B, N)

        # Spearman Loss (Eq. 4)
        # Get ranks of matched keypoints using boolean indexing
        matched_ranks_a = soft_ranks_a_norm[
            mutual_mask_a
        ]  # (M,) where M = total matches across batch

        if len(matched_ranks_a) > 0:
            # Get corresponding ranks in view B
            batch_idx = (
                torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
            )  # (B, N)
            matched_idx_b = nearest_idx_a[mutual_mask_a]  # (M,)
            matched_ranks_b = soft_ranks_b_norm[
                batch_idx[mutual_mask_a], matched_idx_b
            ]  # (M,)

            # MSE loss on normalized ranks
            spearman_loss = F.mse_loss(matched_ranks_a, matched_ranks_b)
        else:
            spearman_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Pull Loss (Eq. 5)
        # Target: matched keypoints → rank 1 (target 0 after normalization)
        #         unmatched keypoints → rank N (target 1 after normalization)
        target_a = torch.where(
            mutual_mask_a,
            torch.zeros_like(soft_ranks_a_norm),  # matched: pull to rank 1 → target 0
            torch.ones_like(soft_ranks_a_norm),  # unmatched: pull to rank N → target 1
        )

        pull_loss = F.l1_loss(soft_ranks_a_norm, target_a)

        total_loss = spearman_loss + self.lambda_ranker * pull_loss

        return total_loss, {
            "spearman_loss": (
                spearman_loss.item() if isinstance(spearman_loss, torch.Tensor) else 0.0
            ),
            "pull_loss": (
                pull_loss.item() if isinstance(pull_loss, torch.Tensor) else 0.0
            ),
            "total_loss": (
                total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0
            ),
        }


class CovarianceLoss(nn.Module):
    """Covariance estimator loss (Eq. 6-7 in paper)."""

    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        covariances_a: torch.Tensor,  # (TotalM, 2, 2)
        covariances_b: torch.Tensor,  # (TotalM, 2, 2)
        reprojection_errors: torch.Tensor,  # (TotalM, 2)
        jacobian: torch.Tensor,  # (TotalM, 2, 2)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute covariance NLL loss on flattened matched keypoints.

        No batch dimension - all matched keypoints across all batches are flattened.
        This is the optimized version that operates only on matched keypoints.

        Args:
            covariances_a: (TotalM, 2, 2) covariance matrices for source keypoints
            covariances_b: (TotalM, 2, 2) covariance matrices for target keypoints
            reprojection_errors: (TotalM, 2) error vectors
            jacobian: (TotalM, 2, 2) Jacobian matrices for homography

        Returns:
            (loss, metrics_dict) tuple
        """
        TotalM = covariances_a.shape[0]
        device = covariances_a.device

        # Early exit if no matches
        if TotalM == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Propagate covariance through homography
        # sigma_b_propagated = J @ cov_b @ J^T
        sigma_a_propagated = torch.einsum(
            "mij,mjk,mlk->mil", jacobian, covariances_a, jacobian
        )

        # Combined error covariance
        sigma_error = covariances_a + sigma_a_propagated  # (TotalM, 2, 2)

        # Add epsilon for numerical stability
        sigma_error = sigma_error + self.epsilon * torch.eye(2, device=device)

        # Ensure positive definite
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(sigma_error)
        except RuntimeError:
            logger.warning("Eigendecomposition failed in forward")
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"log_det": zero, "mahalanobis": zero, "total_points": TotalM}

        eigenvalues = torch.clamp(eigenvalues, min=self.epsilon)

        sigma_inv = torch.einsum(
            "...ij,...j,...kj->...ik", eigenvectors, 1.0 / eigenvalues, eigenvectors
        )

        log_det = 2 * torch.log(eigenvectors).sum(dim=-1)  # (TotalM,)

        mahalanobis = torch.einsum(
            "mi,mij,mj->m", reprojection_errors, sigma_inv, reprojection_errors
        )  # (TotalM,)

        if not torch.isfinite(mahalanobis).all():
            logger.warning("NaN/Inf in mahalanobis")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Negative log-likelihood
        nll = 0.5 * (log_det + mahalanobis)  # (TotalM,)

        nll = nll.clamp(max=100.0)

        # Final check
        finite_mask = torch.isfinite(nll)
        if not finite_mask.all():
            nll = nll[finite_mask]
            if nll.numel():
                logger.warning("bad nll in forward")
                zero = torch.tensor(0.0, device=device, requires_grad=True)
                return zero, {
                    "log_det": zero,
                    "mahalanobis": zero,
                    "total_points": TotalM,
                }

        # Average over all matched keypoints
        loss = nll.mean()

        return loss, {
            "log_det": log_det,
            "mahalanobis": mahalanobis,
            "total_points": TotalM,
        }

    def forward_bidirectional(
        self,
        covariances_a: torch.Tensor,
        covariances_b: torch.Tensor,
        errors_b_to_a: torch.Tensor,
        errors_a_to_b: torch.Tensor,
        jacobian_b_to_a: torch.Tensor,
        jacobian_a_to_b: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        valid_mask_a_to_b: Optional[torch.Tensor] = None,
        valid_mask_b_to_a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute bidirectional covariance loss.

        Args:
            covariances_a: (B, N, 2, 2) covariance for keypoints in view A
            covariances_b: (B, N, 2, 2) covariance for keypoints in view B
            errors_b_to_a: (B, N, 2) reprojection error from B to A
            errors_a_to_b: (B, N, 2) reprojection error from A to B
            jacobian_b_to_a: (B, N, 2, 2) Jacobian of homography B->A
            jacobian_a_to_b: (B, N, 2, 2) Jacobian of homography A->B
            valid_mask: Deprecated, use valid_mask_a_to_b/valid_mask_b_to_a
            valid_mask_a_to_b: (B, N) mask for A->B direction (mutual_match from 0 to 1)
            valid_mask_b_to_a: (B, N) mask for B->A direction (mutual_match from 1 to 0)
        """
        # Support both deprecated single mask and new dual mask
        mask_a_to_b = valid_mask_a_to_b if valid_mask_a_to_b is not None else valid_mask
        mask_b_to_a = valid_mask_b_to_a if valid_mask_b_to_a is not None else valid_mask

        loss_b_to_a = self.forward(
            covariances_a, covariances_b, errors_b_to_a, jacobian_b_to_a, mask_b_to_a
        )
        loss_a_to_b = self.forward(
            covariances_b, covariances_a, errors_a_to_b, jacobian_a_to_b, mask_a_to_b
        )
        total_loss = (loss_b_to_a + loss_a_to_b) / 2

        return total_loss, {
            "cov_nll_b_to_a": (
                loss_b_to_a.item() if isinstance(loss_b_to_a, torch.Tensor) else 0.0
            ),
            "cov_nll_a_to_b": (
                loss_a_to_b.item() if isinstance(loss_a_to_b, torch.Tensor) else 0.0
            ),
            "cov_total": (
                total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0
            ),
        }
