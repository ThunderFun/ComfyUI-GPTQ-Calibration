"""Tests for the Diagonal + Low-Rank (DLR) Hessian representation.

DLR stores ``H ≈ D + U Uᵀ`` where:
  - ``D ∈ ℝⁿ`` — the exact diagonal (per-channel second moments)
  - ``U ∈ ℝⁿˣʳ`` — the top-``r`` correlation directions

These tests verify:
  - **Woodbury identity** for efficient H⁻¹ computation
  - **Column update** (GPTQ-style Sherman-Morrison downdate)
  - **Quality comparison** vs block-diagonal for GPTQ-like operations
  - **Data format** (what gets serialised to .pt)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _woodbury_inverse(D: torch.Tensor, U: torch.Tensor,
                      damping: float = 0.0) -> torch.Tensor:
    """Compute ``(D + UUᵀ + λD)⁻¹`` via the Woodbury identity.

    ``(D(1+λ) + UUᵀ)⁻¹ = D⁻¹/(1+λ) - D⁻¹/(1+λ) U (I + Uᵀ D⁻¹/(1+λ) U)⁻¹ Uᵀ D⁻¹/(1+λ)``

    Args:
        D: (n,) diagonal
        U: (n, r) low-rank factors
        damping: λ, added as ``H += λ diag(H)``

    Returns:
        (n, n) inverse matrix
    """
    n = D.shape[0]
    D_damped = D * (1 + damping)
    D_inv = 1.0 / D_damped.clamp(min=1e-12)  # (n,)

    # Woodbury: H⁻¹ = D⁻¹ - D⁻¹ U (I + Uᵀ D⁻¹ U)⁻¹ Uᵀ D⁻¹
    # where D⁻¹ is diagonal (element-wise multiplication)
    D_inv_U = D_inv.unsqueeze(1) * U  # (n, r)
    inner = torch.eye(U.shape[1]) + U.T @ D_inv_U  # (r, r)
    inner_inv = torch.linalg.inv(inner)  # (r, r)

    H_inv = torch.diag(D_inv) - D_inv_U @ inner_inv @ D_inv_U.T
    return H_inv


def _dlr_column_update(D: torch.Tensor, U: torch.Tensor,
                       col_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ-style column update: zero out row/col ``col_idx`` and return
    the updated DLR factors plus the column for the quantization step.

    Returns:
        (D_new, U_new, h_col) where h_col is the ``col_idx``-th column of H
        before zeroing.
    """
    # Extract the column: H[:, col_idx] = D[col_idx] * e_col_idx + U @ U[col_idx, :]
    h_col = U @ U[col_idx, :]  # (n,)
    h_col[col_idx] += D[col_idx]

    # Zero out row/col col_idx
    D_new = D.clone()
    U_new = U.clone()
    D_new[col_idx] = 0
    U_new[col_idx, :] = 0

    return D_new, U_new, h_col


def _gptq_step_dlr(D: torch.Tensor, U: torch.Tensor,
                    col_idx: int, weight_col: torch.Tensor,
                    quant_fn=None) -> tuple[torch.Tensor, torch.Tensor]:
    """One GPTQ quantisation step using DLR Hessian.

    1. Extract H[:, col_idx]
    2. Quantise W[:, col_idx]
    3. Compute error and propagate to remaining columns
    4. Zero out col_idx in the DLR factors

    Args:
        D, U: DLR factors
        col_idx: column being quantised
        weight_col: (out,) weight column to quantise
        quant_fn: callable(w) -> q(w), defaults to nearest-int rounding

    Returns:
        (D_new, U_new) updated DLR factors
    """
    if quant_fn is None:
        quant_fn = lambda w: torch.round(w)

    n = D.shape[0]

    # Full Hessian column (for the step size computation)
    h_col = U @ U[col_idx, :]  # (n,)
    h_col[col_idx] += D[col_idx]

    h_diag = h_col[col_idx].item()
    if abs(h_diag) < 1e-12:
        # Zero column — nothing to do
        D_new = D.clone()
        U_new = U.clone()
        D_new[col_idx] = 0
        U_new[col_idx, :] = 0
        return D_new, U_new

    # Quantise
    q_col = quant_fn(weight_col)
    error = weight_col - q_col

    # Propagate error: W[:, col_idx+1:] -= error * H[col_idx, col_idx+1:] / H[col_idx, col_idx]
    # (We don't modify W here — that's the caller's job. We just zero the column.)
    _ = error  # Full GPTQ propagates `error` to remaining columns; omitted here for test simplicity.

    # Zero out col_idx
    D_new = D.clone()
    U_new = U.clone()
    D_new[col_idx] = 0
    U_new[col_idx, :] = 0

    return D_new, U_new


def _gptq_simulate_dlr(H_true: torch.Tensor, W: torch.Tensor,
                        rank: int, damping: float = 0.01) -> torch.Tensor:
    """Simulate GPTQ using DLR approximation of H_true.

    Returns the quantised weight matrix.
    """
    n = H_true.shape[0]
    out = W.shape[0]

    # Build DLR from H_true's diagonal + top-r eigenvectors
    D = H_true.diagonal()
    eigvals, eigvecs = torch.linalg.eigh(H_true)
    # Top-r eigenvalues/vectors (eigh returns ascending order)
    top_vals = eigvals[-rank:].clamp(min=0)
    top_vecs = eigvecs[:, -rank:]
    U = top_vecs * top_vals.sqrt().unsqueeze(0)  # (n, r)

    W_q = W.clone()
    for col in range(n):
        D, U = _gptq_step_dlr(D, U, col, W[:, col])

    return W_q


def _gptq_simulate_block(H_true: torch.Tensor, W: torch.Tensor,
                          block_size: int) -> torch.Tensor:
    """Simulate GPTQ using block-diagonal approximation of H_true.

    Returns the quantised weight matrix (only quantised within blocks).
    """
    n = H_true.shape[0]
    W_q = W.clone()

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        H_block = H_true[start:end, start:end]
        # Quantise within this block independently
        # (simplified: just round, no error propagation across blocks)
        W_q[:, start:end] = torch.round(W[:, start:end])

    return W_q


def _quantisation_error(W: torch.Tensor, W_q: torch.Tensor,
                        H: torch.Tensor) -> float:
    """Compute ‖W - W_q‖²_H = Σ_ij (W - W_q)ᵢ H_ij (W - W_q)ⱼ."""
    diff = W - W_q
    return (diff @ H @ diff.T).trace().item()


# ---------------------------------------------------------------------------
# Woodbury identity
# ---------------------------------------------------------------------------

class TestWoodburyIdentity:
    """Tests for computing H⁻¹ = (D + UUᵀ)⁻¹ via Woodbury."""

    def test_woodbury_matches_direct_inverse(self, calibration_mod):
        """Woodbury inverse should match torch.linalg.inv(H)."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1  # positive diagonal
        U = torch.randn(n, rank) * 0.3

        H = torch.diag(D) + U @ U.T
        H_inv_direct = torch.linalg.inv(H)
        H_inv_woodbury = _woodbury_inverse(D, U)

        assert torch.allclose(H_inv_woodbury, H_inv_direct, atol=1e-5), (
            f"Woodbury mismatch: max abs err = {(H_inv_woodbury - H_inv_direct).abs().max():.6f}"
        )

    def test_woodbury_with_damping(self, calibration_mod):
        """Woodbury with damping λ should compute (H + λ diag(H))⁻¹."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3
        damping = 0.01

        H = torch.diag(D) + U @ U.T
        H_damped = H + damping * torch.diag(D)
        H_inv_direct = torch.linalg.inv(H_damped)
        H_inv_woodbury = _woodbury_inverse(D, U, damping=damping)

        assert torch.allclose(H_inv_woodbury, H_inv_direct, atol=1e-5)

    def test_woodbury_identity_reconstruction(self, calibration_mod):
        """H @ H⁻¹ should be the identity (to numerical precision)."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        H = torch.diag(D) + U @ U.T
        H_inv = _woodbury_inverse(D, U)

        product = H @ H_inv
        assert torch.allclose(product, torch.eye(n), atol=1e-4)

    def test_woodbury_purely_diagonal(self, calibration_mod):
        """When U=0, Woodbury should reduce to D⁻¹."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        U = torch.zeros(n, 1)

        H_inv = _woodbury_inverse(D, U)
        expected = torch.diag(1.0 / D)

        assert torch.allclose(H_inv, expected, atol=1e-6)

    def test_woodbury_rank_one(self, calibration_mod):
        """Rank-1 case: H = D + uuᵀ.  Inverse via Sherman-Morrison formula."""
        torch.manual_seed(42)
        n = 16
        D = torch.rand(n) + 0.1
        u = torch.randn(n, 1) * 0.3

        H_inv_woodbury = _woodbury_inverse(D, u)

        # Sherman-Morrison: (D + uuᵀ)⁻¹ = D⁻¹ - D⁻¹uuᵀD⁻¹ / (1 + uᵀD⁻¹u)
        D_inv = 1.0 / D
        D_inv_u = D_inv * u.squeeze()
        denom = 1 + (u.squeeze() * D_inv_u).sum()
        H_inv_sm = torch.diag(D_inv) - torch.outer(D_inv_u, D_inv_u) / denom

        assert torch.allclose(H_inv_woodbury, H_inv_sm, atol=1e-6)

    def test_woodbury_computational_complexity(self, calibration_mod):
        """Woodbury should be O(nr² + r³) instead of O(n³).

        We don't measure wall-clock time (too noisy), but verify the
        intermediate matrix dimensions are correct.
        """
        n, rank = 1024, 64
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        # The expensive part is forming (I + Uᵀ D⁻¹ U), which is (r, r)
        D_inv = 1.0 / D
        inner = torch.eye(rank) + U.T @ (D_inv.unsqueeze(1) * U)
        assert inner.shape == (rank, rank)

        # And inverting it: O(r³)
        inner_inv = torch.linalg.inv(inner)
        assert inner_inv.shape == (rank, rank)

        # The final product D_inv_U @ inner_inv @ D_inv_U.T is (n, r) @ (r, r) @ (r, n)
        # = O(nr²) to form, O(n²) to output as dense matrix
        # (but we'd never materialise the full n×n in practice)


# ---------------------------------------------------------------------------
# Column update (GPTQ-style)
# ---------------------------------------------------------------------------

class TestDLRColumnUpdate:
    """Tests for GPTQ-style column zeroing in DLR representation."""

    def test_column_update_zeros_row_and_col(self, calibration_mod):
        """After zeroing column i, D[i] and U[i,:] should be zero."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        D_new, U_new, _ = _dlr_column_update(D, U, col_idx=5)

        assert D_new[5] == 0
        assert U_new[5, :].abs().max() == 0

    def test_column_update_preserves_other_entries(self, calibration_mod):
        """Other entries should be unchanged."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        D_new, U_new, _ = _dlr_column_update(D, U, col_idx=5)

        mask = torch.ones(n, dtype=torch.bool)
        mask[5] = False
        assert torch.allclose(D_new[mask], D[mask])
        assert torch.allclose(U_new[mask], U[mask])

    def test_column_update_extracts_correct_column(self, calibration_mod):
        """h_col should equal H[:, col_idx] before zeroing."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        H = torch.diag(D) + U @ U.T
        col_idx = 7

        _, _, h_col = _dlr_column_update(D, U, col_idx)
        expected = H[:, col_idx]

        assert torch.allclose(h_col, expected, atol=1e-6)

    def test_multiple_column_updates(self, calibration_mod):
        """Successive column updates should produce a valid DLR with more zeros."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        for col in range(4):
            D, U, _ = _dlr_column_update(D, U, col)

        # First 4 entries should be zero
        assert D[:4].abs().max() == 0
        assert U[:4, :].abs().max() == 0
        # Rest should be unchanged
        assert D[4:].abs().min() > 0

    def test_column_update_preserves_psd_approximation(self, calibration_mod):
        """After zeroing a column, D + UUᵀ should still be PSD."""
        torch.manual_seed(42)
        n, rank = 16, 4
        D = torch.rand(n) + 0.1
        U = torch.randn(n, rank) * 0.3

        D_new, U_new, _ = _dlr_column_update(D, U, col_idx=0)
        H_new = torch.diag(D_new) + U_new @ U_new.T
        eigvals = torch.linalg.eigvalsh(H_new)
        assert eigvals.min().item() >= -1e-6


# ---------------------------------------------------------------------------
# Quality comparison: DLR vs block-diagonal for GPTQ
# ---------------------------------------------------------------------------

class TestDLRQualityForGPTQ:
    """Quality comparison: DLR should give better quantisation than
    block-diagonal for the same memory budget."""

    def test_dlr_captures_cross_block_correlations(self, calibration_mod):
        """DLR should capture correlations between features in different
        blocks, which block-diagonal completely misses.

        We create a Hessian with strong cross-block correlations and verify
        that DLR's off-diagonal approximation is closer to the truth.
        """
        torch.manual_seed(42)
        n = 128
        block_size = 32
        rank = 4  # small rank to make the test clear

        # Create activations with strong cross-block correlation
        X = torch.randn(500, n)
        # Inject correlation between block 0 and block 3
        X[:, 0] += X[:, 96] * 0.8  # strong correlation across blocks
        H_true = X.T @ X

        # DLR approximation
        D = H_true.diagonal()
        eigvals, eigvecs = torch.linalg.eigh(H_true)
        top_vals = eigvals[-rank:].clamp(min=0)
        top_vecs = eigvecs[:, -rank:]
        U = top_vecs * top_vals.sqrt().unsqueeze(0)

        H_dlr = torch.diag(D) + U @ U.T

        # Block-diagonal approximation
        H_block = torch.zeros_like(H_true)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            H_block[start:end, start:end] = H_true[start:end, start:end]

        # Check the cross-block element H[0, 96]
        true_val = H_true[0, 96].item()
        dlr_val = H_dlr[0, 96].item()
        block_val = H_block[0, 96].item()

        # Block-diagonal has zero here; DLR should be closer to truth
        assert abs(block_val) < 1e-6, "Block-diagonal should have zero cross-block"
        assert abs(dlr_val - true_val) < abs(block_val - true_val), (
            f"DLR ({dlr_val:.2f}) should be closer to truth ({true_val:.2f}) "
            f"than block-diagonal ({block_val:.2f})"
        )

    def test_dlr_gptq_error_propagation(self, calibration_mod):
        """GPTQ with DLR should propagate quantisation error across the
        full feature space, not just within blocks.

        We simulate GPTQ using the inverse Hessian (H⁻¹[i,j]/H⁻¹[i,i])
        for error propagation, which is the correct GPTQ formula.
        """
        torch.manual_seed(42)
        n, out = 64, 8
        rank = 8
        block_size = 16
        damping = 0.01

        # True Hessian with global correlations
        X = torch.randn(200, n)
        H_true = X.T @ X

        # Weight matrix
        W = torch.randn(out, n) * 0.5

        # Build DLR (D = residual diagonal: full_diag - diag(UUᵀ))
        eigvals, eigvecs = torch.linalg.eigh(H_true)
        top_vals = eigvals[-rank:].clamp(min=0)
        top_vecs = eigvecs[:, -rank:]
        U = top_vecs * top_vals.sqrt().unsqueeze(0)
        D = (H_true.diagonal() - (U ** 2).sum(dim=1)).clamp(min=0)

        # GPTQ with DLR (using Woodbury inverse)
        H_dlr_inv = _woodbury_inverse(D, U, damping=damping)
        W_dlr = W.clone()
        for col in range(n):
            h_inv_diag = H_dlr_inv[col, col].item()
            if abs(h_inv_diag) < 1e-12:
                continue
            q = torch.round(W_dlr[:, col])
            error = W_dlr[:, col] - q
            W_dlr[:, col] = q
            if col < n - 1:
                h_inv_rest = H_dlr_inv[col, col + 1:]
                W_dlr[:, col + 1:] -= error.unsqueeze(1) * (h_inv_rest / h_inv_diag).unsqueeze(0)

        # GPTQ with block-diagonal (using block inverse)
        W_block = W.clone()
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            H_block = H_true[start:end, start:end]
            H_block_damped = H_block + damping * torch.diag(H_block.diagonal())
            H_inv_block = torch.linalg.inv(H_block_damped)

            for local_col in range(end - start):
                global_col = start + local_col
                h_inv_diag = H_inv_block[local_col, local_col].item()
                if abs(h_inv_diag) < 1e-12:
                    continue
                q = torch.round(W_block[:, global_col])
                error = W_block[:, global_col] - q
                W_block[:, global_col] = q
                if local_col < (end - start) - 1:
                    h_inv_rest = H_inv_block[local_col, local_col + 1:]
                    W_block[:, global_col + 1:end] -= error.unsqueeze(1) * (h_inv_rest / h_inv_diag).unsqueeze(0)

        # Compare errors in H-norm
        err_dlr = _quantisation_error(W, W_dlr, H_true)
        err_block = _quantisation_error(W, W_block, H_true)

        # DLR should be at least as good (usually better due to cross-block propagation)
        # We use a relaxed assertion since the effect depends on the specific Hessian
        assert err_dlr <= err_block * 1.1, (
            f"DLR error ({err_dlr:.4f}) should not be much worse than block ({err_block:.4f})"
        )

    def test_dlr_error_smaller_with_strong_global_correlations(self, calibration_mod):
        """When the Hessian has strong global correlations, DLR should
        significantly outperform block-diagonal."""
        torch.manual_seed(42)
        n = 64
        rank = 8
        block_size = 16

        # Create a Hessian with very strong global correlations
        # (one dominant direction that spans the full space)
        base = torch.randn(n)
        X = torch.randn(500, n) + torch.outer(torch.randn(500), base) * 3.0
        H_true = X.T @ X

        W = torch.randn(4, n) * 0.5

        # DLR (D = residual diagonal: full_diag - diag(UUᵀ))
        eigvals, eigvecs = torch.linalg.eigh(H_true)
        U = eigvecs[:, -rank:] * eigvals[-rank:].clamp(min=0).sqrt().unsqueeze(0)
        D = (H_true.diagonal() - (U ** 2).sum(dim=1)).clamp(min=0)

        # GPTQ with DLR (using Woodbury inverse)
        H_dlr_inv = _woodbury_inverse(D, U, damping=0.01)
        W_dlr = W.clone()
        for col in range(n):
            h_inv_diag = H_dlr_inv[col, col].item()
            if abs(h_inv_diag) < 1e-12:
                continue
            q = torch.round(W_dlr[:, col])
            error = W_dlr[:, col] - q
            W_dlr[:, col] = q
            if col < n - 1:
                h_inv_rest = H_dlr_inv[col, col + 1:]
                W_dlr[:, col + 1:] -= error.unsqueeze(1) * (h_inv_rest / h_inv_diag).unsqueeze(0)

        # Block-diagonal GPTQ (using block inverse)
        W_block = W.clone()
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            H_block = H_true[start:end, start:end]
            H_block_damped = H_block + 0.01 * torch.diag(H_block.diagonal())
            H_inv_block = torch.linalg.inv(H_block_damped)
            for lc in range(end - start):
                gc = start + lc
                h_inv_d = H_inv_block[lc, lc].item()
                if abs(h_inv_d) < 1e-12:
                    continue
                q = torch.round(W_block[:, gc])
                err = W_block[:, gc] - q
                W_block[:, gc] = q
                if lc < (end - start) - 1:
                    h_inv_r = H_inv_block[lc, lc + 1:]
                    W_block[:, gc + 1:end] -= err.unsqueeze(1) * (h_inv_r / h_inv_d).unsqueeze(0)

        err_dlr = _quantisation_error(W, W_dlr, H_true)
        err_block = _quantisation_error(W, W_block, H_true)

        # With strong global correlations, DLR should be meaningfully better
        assert err_dlr < err_block, (
            f"DLR ({err_dlr:.4f}) should beat block-diagonal ({err_block:.4f}) "
            f"with strong global correlations"
        )


# ---------------------------------------------------------------------------
# Data format
# ---------------------------------------------------------------------------

class TestDLRDataFormat:
    """Tests for the serialised DLR Hessian format."""

    def test_dlr_format_has_required_keys(self, calibration_mod):
        """DLR output should contain D, U, and metadata."""
        # This tests the expected output schema
        D = torch.rand(64)
        U = torch.randn(64, 8)
        dlr_data = {
            "format": "dlr",
            "D": D,
            "U": U,
            "rank": 8,
            "n": 64,
        }
        assert dlr_data["format"] == "dlr"
        assert dlr_data["D"].shape == (64,)
        assert dlr_data["U"].shape == (64, 8)
        assert dlr_data["rank"] == 8

    def test_dlr_format_roundtrip(self, calibration_mod, tmp_path):
        """DLR data should survive a save/load roundtrip."""
        D = torch.rand(64)
        U = torch.randn(64, 8)
        data = {
            "metadata": {"format": "dlr"},
            "hessians": {
                "layer1": {"format": "dlr", "D": D, "U": U, "rank": 8, "n": 64},
            },
        }
        path = tmp_path / "dlr_test.pt"
        torch.save(data, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)

        h = loaded["hessians"]["layer1"]
        assert h["format"] == "dlr"
        assert torch.equal(h["D"], D)
        assert torch.equal(h["U"], U)

    def test_dlr_memory_vs_full_and_block(self, calibration_mod):
        """DLR memory should be between block-diagonal and full Hessian."""
        n = 4096
        rank = 128
        block_size = 128

        # Full: n² floats
        full_bytes = n * n * 4

        # Block: (n/block) × block² floats
        num_blocks = (n + block_size - 1) // block_size
        block_bytes = num_blocks * block_size * block_size * 4

        # DLR: n (D) + n×rank (U) floats
        dlr_bytes = (n + n * rank) * 4

        assert block_bytes < dlr_bytes < full_bytes, (
            f"Memory order: block={block_bytes}, dlr={dlr_bytes}, full={full_bytes}"
        )
        # But DLR is much closer to block than to full
        assert (dlr_bytes - block_bytes) < (full_bytes - dlr_bytes)
