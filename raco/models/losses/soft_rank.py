"""
Fast Differentiable Sorting and Ranking (simplified implementation).
Adapted from https://github.com/google-research/fast-soft-sort

Reference:
Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
"Fast Differentiable Sorting and Ranking" (2020)
https://arxiv.org/abs/2002.08871

This simplified version:
- Removes dependency on numba and scipy
- Only implements soft_rank with L2 regularization
- Optimized for the RaCo project's needs
"""

import numpy as np
import torch
from typing import Tuple


# =============================================================================
# Isotonic Regression (PAV Algorithm) - Pure Python Implementation
# =============================================================================

def isotonic_regression_l2(y: np.ndarray) -> np.ndarray:
    """
    Solves isotonic regression: argmin_{v_1 >= ... >= v_n} 0.5||v - y||^2

    Pure Python implementation of PAV (Pool Adjacent Violators) algorithm.
    Time complexity: O(n)

    This implementation follows scikit-learn's PAV algorithm adapted for
    monotonically decreasing output.

    Args:
        y: Input array [n]

    Returns:
        Solution array [n] (monotonically decreasing)
    """
    n = y.shape[0]
    target = np.arange(n)
    c = np.ones(n)
    sums = np.zeros(n)
    solution = np.zeros(n)

    # target describes a list of blocks. At any time, if [i..j] (inclusive) is
    # an active block, then target[i] := j and target[j] := i.

    for i in range(n):
        solution[i] = y[i]
        sums[i] = y[i]

    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if solution[i] > solution[k]:
            i = k
            continue
        sum_y = sums[i]
        sum_c = c[i]
        while True:
            # We are within an increasing subsequence.
            prev_y = solution[k]
            sum_y += sums[k]
            sum_c += c[k]
            k = target[k] + 1
            if k == n or prev_y > solution[k]:
                # Non-singleton increasing subsequence is finished,
                # update first entry.
                solution[i] = sum_y / sum_c
                sums[i] = sum_y
                c[i] = sum_c
                target[i] = k - 1
                target[k - 1] = i
                if i > 0:
                    # Backtrack if we can. This makes the algorithm
                    # single-pass and ensures O(n) complexity.
                    i = target[i - 1]
                break
    return solution


# =============================================================================
# Helper Functions
# =============================================================================

def _partition(solution: np.ndarray, eps: float = 1e-9):
    """Returns partition corresponding to isotonic solution.

    Groups consecutive equal values in solution.
    """
    if len(solution) == 0:
        return []

    sizes = [1]
    for i in range(1, len(solution)):
        if abs(solution[i] - solution[i - 1]) > eps:
            sizes.append(0)
        sizes[-1] += 1

    return sizes


def _inv_permutation(permutation: np.ndarray) -> np.ndarray:
    """Returns inverse permutation."""
    inv_permutation = np.zeros(len(permutation), dtype=int)
    inv_permutation[permutation] = np.arange(len(permutation))
    return inv_permutation


# =============================================================================
# Soft Rank Computation (Numpy Backend)
# =============================================================================

class _SoftRankNumpy:
    """
    Compute soft ranks using isotonic regression.

    This is the numpy backend for soft_rank computation.
    Corresponds to combination of Projection and SoftRank in numpy_ops.py
    """

    def __init__(self, values: np.ndarray, regularization_strength: float = 1.0):
        """
        Args:
            values: Input values to rank [N]
            regularization_strength: Regularization parameter (smaller = closer to true ranks)
        """
        self.values = values
        self.regularization_strength = regularization_strength
        self.scale = 1.0 / regularization_strength

        # Target weights for permutahedron (descending ranks)
        self.input_w = np.arange(len(values))[::-1] + 1

        # Cached computation results
        self.permutation_ = None
        self.inv_permutation_ = None
        self.dual_sol_ = None
        self.primal_sol_ = None

    def compute(self) -> np.ndarray:
        """
        Compute soft ranks using isotonic regression.

        Returns:
            Soft rank values [N] (approximation of true ranks)
        """
        # Scale input values
        scaled_values = self.values * self.scale

        # Sort in descending order
        self.permutation_ = np.argsort(scaled_values)[::-1]
        input_s = scaled_values[self.permutation_]

        # Solve isotonic regression: argmin_{v_1 >= ... >= v_n} 0.5||v - (s-w)||^2
        # This gives us the dual solution
        self.dual_sol_ = isotonic_regression_l2(input_s - self.input_w)

        # Recover primal solution: projection onto permutahedron
        self.primal_sol_ = input_s - self.dual_sol_

        # Inverse permutation to get back to original order
        self.inv_permutation_ = _inv_permutation(self.permutation_)

        return self.primal_sol_[self.inv_permutation_]

    def vjp(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Vector-Jacobian product for backpropagation.

        Args:
            grad_output: Gradient w.r.t. output [N]

        Returns:
            Gradient w.r.t. input values [N]
        """
        # Compute gradient through the permutation and projection
        # The gradient comes from the isotonic regression VJP
        ret = grad_output.copy()

        # Map gradient through permutation
        grad_permuted = grad_output[self.permutation_]

        # Compute VJP for isotonic regression
        # In L2 case, gradient is averaged within each partition block
        grad_dual = np.zeros_like(self.dual_sol_)
        start = 0
        for size in _partition(self.dual_sol_):
            end = start + size
            # L2 regularization: gradient is average within block
            grad_block = grad_permuted[start:end]
            # Contribution from primal solution: primal = s - dual
            # d_primal/d_dual = -1, d_primal/d_s = 1
            # We need to propagate gradient through s
            avg_grad = np.mean(grad_block)
            grad_dual[start:end] = avg_grad
            start = end

        # Subtract because primal = s - dual
        grad_s = grad_permuted - grad_dual

        # Map back through permutation
        ret = grad_s[self.inv_permutation_]

        # Scale gradient
        ret *= self.scale

        return ret


# =============================================================================
# PyTorch Autograd Function
# =============================================================================

class SoftRankFunction(torch.autograd.Function):
    """PyTorch autograd function for soft ranking."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, regularization_strength: float) -> torch.Tensor:
        """
        Forward pass: compute soft ranks.

        Args:
            values: Input tensor [B, N]
            regularization_strength: Regularization parameter

        Returns:
            Soft ranks tensor [B, N]
        """
        # Convert to numpy for computation
        device = values.device
        dtype = values.dtype
        values_np = values.detach().cpu().numpy()

        # Compute soft ranks for each batch
        batch_size = values_np.shape[0]
        ranks_list = []
        objects_list = []

        for i in range(batch_size):
            obj = _SoftRankNumpy(values_np[i], regularization_strength)
            ranks = obj.compute()
            ranks_list.append(ranks)
            objects_list.append(obj)

        # Stack results
        ranks_np = np.stack(ranks_list, axis=0)

        # Save for backward
        ctx.objects_list = objects_list

        return torch.from_numpy(ranks_np).to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: compute gradients.

        Args:
            grad_output: Gradient w.r.t. output [B, N]

        Returns:
            Gradient w.r.t. input values [B, N], None for regularization_strength
        """
        # Convert gradient to numpy
        grad_np = grad_output.detach().cpu().numpy()

        # Compute VJP for each batch
        grad_input_list = []
        for i, obj in enumerate(ctx.objects_list):
            grad_input = obj.vjp(grad_np[i])
            grad_input_list.append(grad_input)

        # Stack gradients
        grad_input_np = np.stack(grad_input_list, axis=0)

        # Convert back to torch tensor
        grad_input = torch.from_numpy(grad_input_np).to(
            device=grad_output.device,
            dtype=grad_output.dtype
        )

        return grad_input, None


# =============================================================================
# Public API
# =============================================================================

def soft_rank(
    values: torch.Tensor,
    direction: str = "ASCENDING",
    regularization_strength: float = 1.0
) -> torch.Tensor:
    """
    Compute differentiable soft ranks.

    The regularization strength determines how close are the returned values
    to the actual ranks. Smaller values = closer to true ranks.

    Args:
        values: [B, N] tensor of values to rank
        direction: "ASCENDING" or "DESCENDING"
        regularization_strength: Regularization parameter (smaller = closer to true ranks)

    Returns:
        [B, N] tensor of soft ranks

    Example:
        >>> values = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
        >>> ranks = soft_rank(values, direction="ASCENDING")
        >>> # ranks ≈ [2, 0, 3, 1] (soft approximation)

    Reference:
        Mathieu Blondel et al., "Fast Differentiable Sorting and Ranking"
        https://arxiv.org/abs/2002.08871
    """
    if len(values.shape) != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")

    if direction not in ("ASCENDING", "DESCENDING"):
        raise ValueError(f"direction should be 'ASCENDING' or 'DESCENDING', got {direction}")

    # For descending order, we negate the values
    if direction == "DESCENDING":
        values = -values

    # Compute soft ranks
    ranks = SoftRankFunction.apply(values, regularization_strength)

    # Negate back if needed
    if direction == "DESCENDING":
        ranks = -ranks

    return ranks
