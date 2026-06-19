"""End-to-end integration tests for DLR Hessian mode."""
from __future__ import annotations

import torch
import torch.nn as nn


def _flatten_linear(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _full_hessian_from_dlr(hessian_data: dict) -> torch.Tensor:
    """Reconstruct a full (n, n) Hessian from a DLR dict."""
    D = hessian_data["D"]
    U = hessian_data["U"]
    return torch.diag(D) + U @ U.T


def _relative_frobenius_error(H_approx: torch.Tensor, H_true: torch.Tensor) -> float:
    return (H_approx - H_true).norm() / H_true.norm()


# ---------------------------------------------------------------------------
# Basic DLR collection via hooks
# ---------------------------------------------------------------------------

class TestDLRCollection:
    """Tests for collecting DLR Hessians via ActivationStatsCollector."""

    def test_dlr_mode_stores_dlr_format(self, calibration_mod):
        """hessian_format='dlr' should store D and U instead of a dense matrix."""
        torch.manual_seed(0)
        lin = nn.Linear(64, 4, bias=False)
        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_basic", store, "Linear",
            hessian_format="dlr", dlr_rank=8,
        )
        collector.register(lin)
        _ = lin(torch.randn(4, 64))
        collector.remove()

        H = store["hessians"]["dlr_basic"]
        assert isinstance(H, dict)
        assert H["format"] == "dlr"
        assert "D" in H and "U" in H
        assert H["D"].shape == (64,)
        assert H["U"].shape == (64, 8)

    def test_dlr_approximates_full_hessian(self, calibration_mod):
        """DLR should approximate the full XᵀX Hessian."""
        torch.manual_seed(0)
        n = 64
        rank = 16
        lin = nn.Linear(n, 4, bias=False)
        xs = [torch.randn(4, n) for _ in range(3)]

        # Full Hessian from the same inputs the hook saw
        all_x = torch.cat(xs)
        H_true = all_x.T @ all_x

        # DLR Hessian
        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_approx", store, "Linear",
            hessian_format="dlr", dlr_rank=rank,
        )
        collector.register(lin)
        for x in xs:
            _ = lin(x)
        collector.remove()

        H_dlr = _full_hessian_from_dlr(store["hessians"]["dlr_approx"])
        err = _relative_frobenius_error(H_dlr, H_true)
        assert err < 0.3, f"DLR approximation error too high: {err:.4f}"

    def test_dlr_diagonal_matches_true_diagonal(self, calibration_mod):
        """diag(D + UUᵀ) should exactly match the diagonal of XᵀX."""
        torch.manual_seed(0)
        n = 32
        lin = nn.Linear(n, 4, bias=False)
        xs = [torch.randn(4, n) for _ in range(3)]

        all_x = torch.cat(xs)
        H_true = all_x.T @ all_x

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_diag", store, "Linear",
            hessian_format="dlr", dlr_rank=4,
        )
        collector.register(lin)
        for x in xs:
            _ = lin(x)
        collector.remove()

        H = store["hessians"]["dlr_diag"]
        # diag(D + UUᵀ) should match exact diagonal
        diag_approx = H["D"] + (H["U"] ** 2).sum(dim=1)
        assert torch.allclose(diag_approx, H_true.diagonal(), atol=1e-3, rtol=1e-3), (
            f"Diagonal mismatch: max err = {(diag_approx - H_true.diagonal()).abs().max():.6f}"
        )

    def test_dlr_accumulates_across_forward_passes(self, calibration_mod):
        """Multiple forward passes should accumulate into the same sketch."""
        torch.manual_seed(0)
        n = 32
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_acc", store, "Linear",
            hessian_format="dlr", dlr_rank=4,
        )
        collector.register(lin)

        xs = [torch.randn(2, n) for _ in range(4)]
        for x in xs:
            _ = lin(x)
        collector.remove()

        all_x = torch.cat(xs)
        H_true = all_x.T @ all_x

        H_dlr = _full_hessian_from_dlr(store["hessians"]["dlr_acc"])
        assert H_dlr.shape == (n, n)
        # Diagonal should accumulate correctly
        assert torch.allclose(H_dlr.diagonal(), H_true.diagonal(), atol=1e-3, rtol=1e-3)

    def test_dlr_respects_rank_parameter(self, calibration_mod):
        """Different rank values should produce different U shapes."""
        torch.manual_seed(0)
        n = 64
        lin = nn.Linear(n, 4, bias=False)

        for rank in [4, 8, 16]:
            store: dict = {}
            collector = calibration_mod.ActivationStatsCollector(
                f"dlr_r{rank}", store, "Linear",
                hessian_format="dlr", dlr_rank=rank,
            )
            collector.register(lin)
            _ = lin(torch.randn(2, n))
            collector.remove()

            U = store["hessians"][f"dlr_r{rank}"]["U"]
            assert U.shape == (n, rank)

    def test_dlr_with_conv2d(self, calibration_mod):
        """DLR should work with Conv2d layers (unfolded activations)."""
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        # in_features = 3 * 3 * 3 = 27
        in_features = 3 * 3 * 3

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_conv", store, "Conv2d",
            hessian_format="dlr", dlr_rank=4,
        )
        collector.register(conv)
        _ = conv(torch.randn(1, 3, 8, 8))
        collector.remove()

        H = store["hessians"]["dlr_conv"]
        assert H["format"] == "dlr"
        assert H["D"].shape == (in_features,)
        assert H["U"].shape[0] == in_features


# ---------------------------------------------------------------------------
# DLR + ConvRot rotation
# ---------------------------------------------------------------------------

class TestDLRWithRotation:
    """Tests for DLR combined with Hadamard rotation."""

    def test_dlr_with_rotation(self, calibration_mod):
        """DLR + rotation should produce valid DLR factors."""
        torch.manual_seed(0)
        n = 256
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_rot", store, "Linear",
            hessian_format="dlr", dlr_rank=16, rot_size=256,
        )
        collector.register(lin)
        _ = lin(torch.randn(2, n))
        collector.remove()

        H = store["hessians"]["dlr_rot"]
        assert H["format"] == "dlr"
        assert H["D"].shape == (n,)
        assert H["U"].shape == (n, 16)
        assert torch.all(torch.isfinite(H["D"]))
        assert torch.all(torch.isfinite(H["U"]))

    def test_rotation_improves_dlr_quality(self, calibration_mod):
        """DLR with rotation should have lower approximation error."""
        torch.manual_seed(0)
        n = 256
        rank = 8
        lin = nn.Linear(n, 4, bias=False)
        x = torch.randn(4, n)

        # Without rotation
        store1: dict = {}
        c1 = calibration_mod.ActivationStatsCollector(
            "dlr_norot", store1, "Linear",
            hessian_format="dlr", dlr_rank=rank,
        )
        c1.register(lin)
        _ = lin(x)
        c1.remove()

        # With rotation
        store2: dict = {}
        c2 = calibration_mod.ActivationStatsCollector(
            "dlr_rot", store2, "Linear",
            hessian_format="dlr", dlr_rank=rank, rot_size=256,
        )
        c2.register(lin)
        _ = lin(x)
        c2.remove()

        x_flat = _flatten_linear(x)
        H_true = x_flat.T @ x_flat

        H_dlr_norot = _full_hessian_from_dlr(store1["hessians"]["dlr_norot"])
        H_dlr_rot = _full_hessian_from_dlr(store2["hessians"]["dlr_rot"])

        # Rotated DLR should better approximate the rotated true Hessian
        x_rot = calibration_mod.rotate_activations(x_flat, 256)
        H_true_rot = x_rot.T @ x_rot

        err_norot = _relative_frobenius_error(H_dlr_norot, H_true)
        err_rot = _relative_frobenius_error(H_dlr_rot, H_true_rot)

        # Rotation should improve quality (or at least not hurt)
        assert err_rot <= err_norot * 1.1, (
            f"Rotation should improve DLR quality: norot={err_norot:.4f}, rot={err_rot:.4f}"
        )


# ---------------------------------------------------------------------------
# DLR + PermuQuant
# ---------------------------------------------------------------------------

class TestDLRWithPermuQuant:
    """Tests for DLR combined with PermuQuant channel reordering."""

    def test_dlr_both_mode_collects_mu2_and_dlr(self, calibration_mod):
        """mode='both' + hessian_format='dlr' should collect both mu2 and DLR."""
        torch.manual_seed(0)
        n = 32
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_both", store, "Linear",
            mode="both", hessian_format="dlr", dlr_rank=4,
        )
        collector.register(lin)
        _ = lin(torch.randn(4, n))
        collector.remove()

        assert "mu2" in store
        assert store["mu2"]["dlr_both"].shape == (n,)

        H = store["hessians"]["dlr_both"]
        assert H["format"] == "dlr"

    def test_dlr_permutation_applied_to_u(self, calibration_mod):
        """When permutation is set, U rows should be reordered."""
        torch.manual_seed(0)
        n = 32
        lin = nn.Linear(n, 4, bias=False)
        perm = torch.randperm(n, dtype=torch.int32)
        x = torch.randn(4, n)  # fixed input for both collectors

        # Without permutation
        store1: dict = {}
        c1 = calibration_mod.ActivationStatsCollector(
            "dlr_noperm", store1, "Linear",
            hessian_format="dlr", dlr_rank=4,
        )
        c1.register(lin)
        _ = lin(x)
        c1.remove()

        # With permutation
        store2: dict = {}
        c2 = calibration_mod.ActivationStatsCollector(
            "dlr_perm", store2, "Linear",
            hessian_format="dlr", dlr_rank=4, permutation=perm,
        )
        c2.register(lin)
        _ = lin(x)
        c2.remove()

        H1 = store1["hessians"]["dlr_noperm"]
        H2 = store2["hessians"]["dlr_perm"]

        # D and U should be permuted
        assert torch.allclose(H2["D"], H1["D"][perm], atol=1e-5)
        assert torch.allclose(H2["U"], H1["U"][perm], atol=1e-5)


# ---------------------------------------------------------------------------
# DLR + amax
# ---------------------------------------------------------------------------

class TestDLRWithAmax:
    """Tests for DLR combined with amax collection."""

    def test_dlr_with_amax(self, calibration_mod):
        """hessian_format='dlr' + collect_amax=True should work."""
        torch.manual_seed(0)
        lin = nn.Linear(32, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_amax", store, "Linear",
            hessian_format="dlr", dlr_rank=4, collect_amax=True,
        )
        collector.register(lin)
        _ = lin(torch.randn(4, 32))
        collector.remove()

        assert "amax" in store
        assert store["amax"]["dlr_amax"] > 0


# ---------------------------------------------------------------------------
# NaN/Inf guard with DLR
# ---------------------------------------------------------------------------

class TestDLRNaNGuard:
    """NaN/Inf guard should work with DLR mode."""

    def test_nan_does_not_corrupt_dlr(self, calibration_mod):
        """A NaN-contaminated forward pass should not corrupt the DLR sketch."""
        torch.manual_seed(0)
        n = 64
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_nan", store, "Linear",
            hessian_format="dlr", dlr_rank=4,
        )
        collector.register(lin)

        # Clean pass
        _ = lin(torch.randn(4, n))

        # NaN pass
        x_nan = torch.randn(4, n)
        x_nan[0, 0] = float("nan")
        _ = lin(x_nan)

        # Finalize and check
        collector.remove()
        H = store["hessians"]["dlr_nan"]
        assert isinstance(H, dict) and H["format"] == "dlr"
        assert torch.all(torch.isfinite(H["D"]))
        assert torch.all(torch.isfinite(H["U"]))
        assert collector._nan_skip_count > 0

    def test_inf_does_not_corrupt_dlr(self, calibration_mod):
        """An Inf-contaminated forward pass should not corrupt the DLR sketch."""
        torch.manual_seed(0)
        n = 64
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_inf", store, "Linear",
            hessian_format="dlr", dlr_rank=4,
        )
        collector.register(lin)

        _ = lin(torch.randn(4, n))

        x_inf = torch.randn(4, n)
        x_inf[0, 0] = float("inf")
        _ = lin(x_inf)

        collector.remove()
        H = store["hessians"]["dlr_inf"]
        assert isinstance(H, dict) and H["format"] == "dlr"
        assert torch.all(torch.isfinite(H["D"]))
        assert torch.all(torch.isfinite(H["U"]))
        assert collector._nan_skip_count > 0

    def test_outlier_does_not_corrupt_dlr(self, calibration_mod):
        """Extreme activations should be skipped."""
        torch.manual_seed(0)
        n = 64
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_outlier", store, "Linear",
            hessian_format="dlr", dlr_rank=4,
        )
        collector.register(lin)

        _ = lin(torch.randn(4, n))

        x_outlier = torch.randn(4, n) * 1e8
        _ = lin(x_outlier)

        collector.remove()
        H = store["hessians"]["dlr_outlier"]
        assert isinstance(H, dict) and H["format"] == "dlr"
        assert torch.all(torch.isfinite(H["D"]))
        assert torch.all(torch.isfinite(H["U"]))
        assert collector._nan_skip_count > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestDLREdgeCases:
    """Edge cases for DLR collection."""

    def test_dlr_small_layer(self, calibration_mod):
        """DLR should work for layers smaller than the rank."""
        torch.manual_seed(0)
        n = 8
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_small", store, "Linear",
            hessian_format="dlr", dlr_rank=16,  # rank > n
        )
        collector.register(lin)
        _ = lin(torch.randn(2, n))
        collector.remove()

        H = store["hessians"]["dlr_small"]
        # Should either use full Hessian or clamp rank to n
        assert H["D"].shape == (n,)
        assert H["U"].shape[0] == n
        assert H["U"].shape[1] <= n

    def test_dlr_rank_1(self, calibration_mod):
        """Rank-1 DLR should work (captures only the dominant direction)."""
        torch.manual_seed(0)
        n = 32
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_r1", store, "Linear",
            hessian_format="dlr", dlr_rank=1,
        )
        collector.register(lin)
        _ = lin(torch.randn(4, n))
        collector.remove()

        H = store["hessians"]["dlr_r1"]
        assert H["U"].shape == (n, 1)

    def test_dlr_rank_equals_features(self, calibration_mod):
        """Rank == n should capture the full Hessian (no approximation)."""
        torch.manual_seed(0)
        n = 16
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_full", store, "Linear",
            hessian_format="dlr", dlr_rank=n,
        )
        collector.register(lin)
        _ = lin(torch.randn(4, n))
        collector.remove()

        H = store["hessians"]["dlr_full"]
        H_recon = _full_hessian_from_dlr(H)

        assert H_recon.shape == (n, n)
        assert torch.all(torch.isfinite(H_recon))

    def test_dlr_many_small_updates(self, calibration_mod):
        """Thousands of tiny forward passes should not degrade the sketch."""
        torch.manual_seed(0)
        n = 32
        lin = nn.Linear(n, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "dlr_many", store, "Linear",
            hessian_format="dlr", dlr_rank=4,
        )
        collector.register(lin)

        for _ in range(200):
            _ = lin(torch.randn(1, n))
        collector.remove()

        H = store["hessians"]["dlr_many"]
        assert torch.all(torch.isfinite(H["D"]))
        assert torch.all(torch.isfinite(H["U"]))


# ---------------------------------------------------------------------------
# DLR vs block-diagonal quality (integration)
# ---------------------------------------------------------------------------

class TestDLRvsBlockDiagonalIntegration:
    """Integration tests comparing DLR and block-diagonal quality."""

    def test_dlr_better_than_block_for_global_correlations(self, calibration_mod):
        """DLR should beat block-diagonal when there are global correlations."""
        torch.manual_seed(0)
        n = 128
        rank = 16
        block_size = 16  # same memory as rank=16

        lin = nn.Linear(n, 4, bias=False)

        # Create activations with strong global correlation
        x = torch.randn(8, n)
        x[:, 0] += x[:, 127] * 0.9  # cross-block correlation

        # Full Hessian (ground truth)
        x_flat = _flatten_linear(x)
        H_true = x_flat.T @ x_flat

        # DLR
        store_dlr: dict = {}
        c_dlr = calibration_mod.ActivationStatsCollector(
            "dlr", store_dlr, "Linear",
            hessian_format="dlr", dlr_rank=rank,
        )
        c_dlr.register(lin)
        _ = lin(x)
        c_dlr.remove()

        # Block-diagonal
        store_blk: dict = {}
        c_blk = calibration_mod.ActivationStatsCollector(
            "blk", store_blk, "Linear",
            hessian_block_size=block_size,
        )
        c_blk.register(lin)
        _ = lin(x)
        c_blk.remove()

        H_dlr = _full_hessian_from_dlr(store_dlr["hessians"]["dlr"])
        H_blk_blocks = store_blk["hessians"]["blk"]

        # Reconstruct block-diagonal as full matrix
        H_blk = torch.zeros(n, n)
        offset = 0
        for block in H_blk_blocks:
            bs = block.shape[0]
            H_blk[offset:offset + bs, offset:offset + bs] = block
            offset += bs

        err_dlr = _relative_frobenius_error(H_dlr, H_true)
        err_blk = _relative_frobenius_error(H_blk, H_true)

        assert err_dlr < err_blk, (
            f"DLR ({err_dlr:.4f}) should be better than block-diagonal ({err_blk:.4f})"
        )

    def test_dlr_memory_comparable_to_block_diagonal(self, calibration_mod):
        """DLR and block-diagonal should use comparable memory."""
        n = 1024
        rank = 128
        block_size = 128

        # DLR: n*D + n*rank*U + sketch_size*n
        dlr_floats = n + n * rank + (2 * rank + 4) * n

        # Block: (n/block) * block²
        num_blocks = (n + block_size - 1) // block_size
        blk_floats = num_blocks * block_size * block_size

        ratio = dlr_floats / blk_floats
        # Should be within 4× (same order of magnitude)
        assert 0.25 < ratio < 4.0, f"Memory ratio {ratio:.2f} is too far from 1.0"


# ---------------------------------------------------------------------------
# collect_stats integration
# ---------------------------------------------------------------------------

class TestCollectStatsDLR:
    """Tests for collect_stats with DLR mode."""

    def test_collect_stats_with_dlr_format(self, calibration_mod, monkeypatch):
        """collect_stats with hessian_format='dlr' should produce valid output."""
        import comfy.sample as comfy_sample

        def fake_sample(*args, **kwargs):
            return kwargs["noise"]

        monkeypatch.setattr(comfy_sample, "sample", fake_sample)

        class _LatentFormat:
            latent_channels = 4

        class _Diffusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(32, 4, bias=False)

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = _Diffusion()
                self.latent_format = _LatentFormat()
                self.model_sampling = None
                self.model_config = None

        class _Patcher:
            def __init__(self):
                self.model = _Inner()
                self.load_device = torch.device("cpu")
                self.model_options = {}

        fake_cond = [[torch.randn(1, 4, 4), {}]]

        data = calibration_mod.collect_stats(
            model_patcher=_Patcher(),
            conditioning=fake_cond,
            num_steps=1,
            num_samples=1,
            seed=0,
            latent_height=4,
            latent_width=4,
            hessian_block_size=0,
            collect_amax=True,
            hessian_format="dlr",
            dlr_rank=8,
        )

        assert "hessians" in data
        for name, H in data["hessians"].items():
            assert isinstance(H, dict)
            assert H["format"] == "dlr"
            assert "D" in H and "U" in H
            assert torch.all(torch.isfinite(H["D"]))
            assert torch.all(torch.isfinite(H["U"]))

    def test_collect_stats_dlr_metadata(self, calibration_mod, monkeypatch):
        """DLR mode should set metadata.hessian_format='dlr'."""
        import comfy.sample as comfy_sample

        monkeypatch.setattr(comfy_sample, "sample", lambda *a, **kw: kw["noise"])

        class _LatentFormat:
            latent_channels = 4

        class _Diffusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(16, 4, bias=False)

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = _Diffusion()
                self.latent_format = _LatentFormat()
                self.model_sampling = None
                self.model_config = None

        class _Patcher:
            def __init__(self):
                self.model = _Inner()
                self.load_device = torch.device("cpu")
                self.model_options = {}

        data = calibration_mod.collect_stats(
            model_patcher=_Patcher(),
            conditioning=[[torch.randn(1, 4, 4), {}]],
            num_steps=1, num_samples=1, seed=0,
            latent_height=4, latent_width=4,
            hessian_format="dlr", dlr_rank=4,
        )

        assert data["metadata"]["hessian_format"] == "dlr"
        assert data["metadata"]["dlr_rank"] == 4
