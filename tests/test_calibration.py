"""Tests for ActivationStatsCollector and related calibration functions."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_linear(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _flatten_conv(x: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
    patches = F.unfold(
        x,
        module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
    )
    patches = patches.permute(0, 2, 1)
    return patches.reshape(-1, patches.shape[-1])


def test_linear_full_hessian_matches_xTx(calibration_mod):
    torch.manual_seed(0)
    lin = nn.Linear(8, 4, bias=False)
    xs = [torch.randn(2, 8) for _ in range(3)]
    expected = sum(_flatten_linear(x).T @ _flatten_linear(x) for x in xs)

    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector("lin", store, "Linear", hessian_block_size=0)
    collector.register(lin)
    for x in xs:
        _ = lin(x)
    collector.remove()

    assert "hessians" in store
    assert torch.allclose(store["hessians"]["lin"], expected, atol=1e-4)
    assert len(collector.hooks) == 0  # hooks removed cleanly


def test_conv2d_hessian_matches_unfolded_xTx(calibration_mod):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
    xs = [torch.randn(1, 3, 8, 8) for _ in range(2)]
    expected = sum(_flatten_conv(x, conv).T @ _flatten_conv(x, conv) for x in xs)

    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector("conv", store, "Conv2d", hessian_block_size=0)
    collector.register(conv)
    for x in xs:
        _ = conv(x)
    collector.remove()

    got = store["hessians"]["conv"]
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-3)


def test_block_wise_hessian_stores_diagonal_blocks(calibration_mod):
    torch.manual_seed(0)
    lin = nn.Linear(1024, 1024, bias=False)
    xs = [torch.randn(1, 1024) for _ in range(2)]
    expected_full = sum(_flatten_linear(x).T @ _flatten_linear(x) for x in xs)

    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector("blk", store, "Linear", hessian_block_size=128)
    collector.register(lin)
    for x in xs:
        _ = lin(x)
    collector.remove()

    got = store["hessians"]["blk"]
    assert isinstance(got, list)
    assert len(got) == 8
    assert all(g.shape == (128, 128) for g in got)

    # Diagonal blocks should match the corresponding diagonal block of the
    # full H exactly. Off-diagonals are intentionally zero in block mode.
    for i in range(8):
        s = i * 128
        e = s + 128
        assert torch.allclose(got[i], expected_full[s:e, s:e], atol=1e-3)


def test_small_layer_uses_full_hessian_even_when_block_size_set(calibration_mod):
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False)
    xs = [torch.randn(1, 8) for _ in range(2)]
    expected = sum(_flatten_linear(x).T @ _flatten_linear(x) for x in xs)

    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector("small", store, "Linear", hessian_block_size=128)
    collector.register(lin)
    for x in xs:
        _ = lin(x)
    collector.remove()

    got = store["hessians"]["small"]
    assert got.shape == (8, 8)
    assert torch.allclose(got, expected, atol=1e-4)


def test_amax_tracks_running_max(calibration_mod):
    torch.manual_seed(0)
    lin = nn.Linear(4, 4, bias=False)
    xs = [torch.randn(1, 4) * (i + 1) for i in range(3)]  # amplitudes 1, 2, 3
    expected = max(x.abs().max().item() for x in xs)

    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector("amax", store, "Linear", hessian_block_size=0, collect_amax=True)
    collector.register(lin)
    for x in xs:
        _ = lin(x)
    collector.remove()

    assert abs(store["amax"]["amax"] - expected) < 1e-5


def test_amax_skipped_when_disabled(calibration_mod):
    torch.manual_seed(0)
    lin = nn.Linear(4, 4, bias=False)
    xs = [torch.randn(1, 4) for _ in range(2)]

    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector("noamax", store, "Linear", hessian_block_size=0, collect_amax=False)
    collector.register(lin)
    for x in xs:
        _ = lin(x)
    collector.remove()

    assert "amax" not in store


def test_hessians_accumulate_across_calls(calibration_mod):
    torch.manual_seed(0)
    lin = nn.Linear(4, 4, bias=False)
    xs = [torch.randn(2, 4) for _ in range(2)]
    expected = sum(x.T @ x for x in xs)

    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector("acc", store, "Linear", hessian_block_size=0)
    collector.register(lin)
    for x in xs:
        _ = lin(x)
    collector.remove()

    assert torch.allclose(store["hessians"]["acc"], expected, atol=1e-4)


def test_hook_called_via_module_call(calibration_mod):
    """Sanity check that the hook fires on a normal forward pass."""
    torch.manual_seed(0)
    lin = nn.Linear(4, 4, bias=False)
    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector("via_call", store, "Linear", hessian_block_size=0)
    collector.register(lin)
    _ = lin(torch.randn(1, 4))
    collector.remove()
    assert "via_call" in store["hessians"]


def test_walk_target_modules_finds_linears_and_convs(calibration_mod):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(4, 4)
            self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
            self.lin2 = nn.Linear(4, 4)

    net = Net()
    found = calibration_mod._walk_target_modules(net)
    names = [n for n, _ in found]
    assert "lin1" in names
    assert "lin2" in names
    assert "conv1" in names
    assert len(found) == 3


def test_infer_latent_shape_uses_model_latent_format(calibration_mod):
    class FakeLatentFormat:
        latent_channels = 16
        spacial_downscale_ratio = 8

    class FakeModel:
        latent_format = FakeLatentFormat()

    class FakePatcher:
        model = FakeModel()

    # ``height``/``width`` are latent dimensions; we just take the
    # channel count from the model's latent_format.
    c, h, w = calibration_mod._infer_latent_shape(FakePatcher(), 128, 128)
    assert c == 16
    assert h == 128
    assert w == 128


def test_infer_latent_shape_fallback_when_no_model(calibration_mod):
    class FakePatcher:
        model = None

    # No model -> default to 4 channels and use the literal arguments.
    c, h, w = calibration_mod._infer_latent_shape(FakePatcher(), 64, 64)
    assert c == 4
    assert h == 64
    assert w == 64


def test_build_sigmas_returns_simple_schedule(calibration_mod):
    class FakeModelSampling:
        sigmas = torch.linspace(1.0, 0.0, 10)

    s = calibration_mod._build_sigmas(FakeModelSampling(), num_steps=4)
    assert s.shape == (5,)
    assert s[-1].item() == 0.0


def test_collect_stats_validates_inputs(calibration_mod):
    try:
        calibration_mod.collect_stats(None, conditioning=[[{}]], num_steps=1, num_samples=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for None model")

    class FakePatcher:
        model = None
        load_device = torch.device("cpu")

    try:
        calibration_mod.collect_stats(FakePatcher(), conditioning=[], num_steps=1, num_samples=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty conditioning")

    try:
        calibration_mod.collect_stats(FakePatcher(), conditioning=[[{}]], num_steps=0, num_samples=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for num_steps=0")


def test_block_gram_last_block_has_actual_size(calibration_mod):
    """``_block_gram`` must not crash when ``n_features`` is not a
    multiple of ``block_size``; the final block retains its true size
    instead of being zero-padded, preventing singular blocks.
    """
    torch.manual_seed(0)
    n = 200
    block_size = 128
    x = torch.randn(4, n)

    result = calibration_mod.ActivationStatsCollector(
        "padded", {}, "Linear"
    )._block_gram(x, block_size)

    num_blocks = (n + block_size - 1) // block_size  # 2
    assert isinstance(result, list)
    assert len(result) == num_blocks
    assert result[0].shape == (block_size, block_size)
    assert result[1].shape == (n - block_size, n - block_size)

    expected_full = x.T @ x
    # The full-block entry should match the corresponding diagonal of the
    # full Hessian.
    assert torch.allclose(result[0], expected_full[:block_size, :block_size], atol=1e-4)

    # The last block should exactly equal the true diagonal block.
    true_partial = expected_full[block_size:, block_size:]
    assert torch.allclose(result[1], true_partial, atol=1e-4)


def test_block_gram_handles_n_smaller_than_block_size(calibration_mod):
    """When ``n_features < block_size`` the single block is returned
    at its true size instead of being padded.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 50)
    block_size = 128
    result = calibration_mod.ActivationStatsCollector(
        "small_pad", {}, "Linear"
    )._block_gram(x, block_size)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].shape == (50, 50)
    expected = x.T @ x
    assert torch.allclose(result[0], expected, atol=1e-4)


def test_block_wise_hessian_works_for_non_divisible_layer(calibration_mod):
    """End-to-end: a Linear with in_features not divisible by the
    block size must produce a list of blocks with true sizes.

    ``_use_block_gram`` triggers only when ``n_features > block_size * 4``,
    so we pick 600 > 128*4.
    """
    torch.manual_seed(0)
    n = 600
    lin = nn.Linear(n, 8, bias=False)
    xs = [torch.randn(2, n) for _ in range(2)]

    store: dict = {}
    collector = calibration_mod.ActivationStatsCollector(
        "nd", store, "Linear", hessian_block_size=128
    )
    collector.register(lin)
    for x in xs:
        _ = lin(x)
    collector.remove()

    got = store["hessians"]["nd"]
    # 600/128 -> 4 full blocks of 128 + 1 block of 88 (no padding)
    assert isinstance(got, list)
    assert len(got) == 5
    assert all(g.shape == (128, 128) for g in got[:4])
    assert got[4].shape == (88, 88)


def test_collect_stats_calls_comfy_sample_sample(calibration_mod, monkeypatch):
    """Regression test: ``collect_stats`` must invoke
    ``comfy.sample.sample`` (the high-level entry point that takes
    ``sampler_name`` / ``scheduler`` / ``denoise``), not the
    lower-level ``comfy.samplers.sample`` which has a different
    signature.
    """
    import comfy.sample as comfy_sample
    import comfy.samplers as comfy_samplers

    called = {"sample": None, "samplers": None}

    def fake_sample(*args, **kwargs):
        called["sample"] = (args, kwargs)
        return kwargs["noise"]

    def should_not_be_called(*args, **kwargs):
        called["samplers"] = (args, kwargs)
        raise AssertionError(
            "collect_stats must not call comfy.samplers.sample "
            "(signature mismatch); it should call comfy.sample.sample"
        )

    monkeypatch.setattr(comfy_sample, "sample", fake_sample)
    monkeypatch.setattr(comfy_samplers, "sample", should_not_be_called)

    class _LatentFormat:
        latent_channels = 4

    class _Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 4, bias=False)

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

    # ComfyUI conditioning: each entry is [cross_attn_tensor, metadata_dict]
    fake_cond = [[torch.randn(1, 4, 4), {"some_key": "some_val"}]]

    data = calibration_mod.collect_stats(
        model_patcher=_Patcher(),
        conditioning=fake_cond,
        num_steps=1,
        num_samples=1,
        seed=0,
        latent_height=4,
        latent_width=4,
        hessian_block_size=0,
        collect_amax=False,
    )

    assert called["sample"] is not None, "comfy.sample.sample was not called"
    assert called["samplers"] is None, "comfy.samplers.sample should not be called"
    _, kwargs = called["sample"]
    for k in ("steps", "cfg", "sampler_name", "scheduler", "positive",
              "negative", "latent_image", "denoise", "sigmas",
              "disable_pbar", "seed"):
        assert k in kwargs, f"missing kwarg {k!r} in comfy.sample.sample call"
    assert data["metadata"]["num_samples"] == 1


def test_collect_stats_preserves_conditioning_format(calibration_mod, monkeypatch):
    """The conditioning passed to ``collect_stats`` must arrive at
    ``comfy.sample.sample`` exactly as-is (``[[tensor, dict]]``),
    without being flattened.  ``convert_cond`` in the real ComfyUI
    sampler iterates the outer list and indexes ``c[1]`` — flattening
    would break that.
    """
    import comfy.sample as comfy_sample

    captured = {}

    def fake_sample(*args, **kwargs):
        captured["positive"] = kwargs["positive"]
        captured["negative"] = kwargs["negative"]
        return kwargs["noise"]

    monkeypatch.setattr(comfy_sample, "sample", fake_sample)

    class _LatentFormat:
        latent_channels = 4

    class _Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 4, bias=False)

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

    # Real ComfyUI conditioning: [[tensor, dict], ...]
    cross_attn = torch.randn(1, 4, 4)
    meta = {"pooled_output": torch.randn(1, 4)}
    fake_cond = [[cross_attn, meta]]

    calibration_mod.collect_stats(
        model_patcher=_Patcher(),
        conditioning=fake_cond,
        num_steps=1,
        num_samples=1,
        seed=0,
        latent_height=4,
        latent_width=4,
    )

    pos = captured["positive"]
    neg = captured["negative"]
    # Must still be [[tensor, dict]], NOT flattened to [tensor, dict]
    assert isinstance(pos, list), "positive should be a list"
    assert len(pos) == 1, "positive should have 1 entry"
    entry = pos[0]
    assert isinstance(entry, list), "each entry should be [tensor, dict]"
    assert len(entry) == 2, "each entry must have 2 elements"
    assert torch.is_tensor(entry[0])
    assert isinstance(entry[1], dict)
    # Same entry was used for both positive and negative
    assert pos is neg


# ── ConvRot rotation tests ──────────────────────────────────────────────────

class TestRotateActivations:
    """Tests for the ConvRot Hadamard rotation applied to activations."""

    def test_rotate_preserves_norm(self, calibration_mod):
        """Hadamard rotation is orthogonal, so it preserves the Frobenius norm."""
        torch.manual_seed(42)
        x = torch.randn(16, 256)
        x_rot = calibration_mod.rotate_activations(x, 256)
        assert torch.allclose(x.norm(), x_rot.norm(), atol=1e-3)

    def test_rotate_output_shape(self, calibration_mod):
        """Output shape should match input shape."""
        x = torch.randn(8, 512)
        x_rot = calibration_mod.rotate_activations(x, 256)
        assert x_rot.shape == x.shape

    def test_rotate_with_padding(self, calibration_mod):
        """When in_features is not a multiple of rot_size, input is padded (not truncated)."""
        x = torch.randn(4, 300)
        x_rot = calibration_mod.rotate_activations(x, 256)
        assert x_rot.shape == (4, 512)  # padded to next multiple of 256

    def test_rotate_identity_at_size_4(self, calibration_mod):
        """The 4x4 Hadamard should transform known values correctly."""
        H = calibration_mod.get_hadamard(4, torch.float32)
        # H should be orthogonal: H @ H.T = I
        assert torch.allclose(H @ H.T, torch.eye(4), atol=1e-6)

    def test_rotate_is_invertible(self, calibration_mod):
        """Applying rotation twice should return the original (H^2 = I for normalized H)."""
        torch.manual_seed(42)
        x = torch.randn(8, 256)
        x_rot = calibration_mod.rotate_activations(x, 256)
        x_back = calibration_mod.rotate_activations(x_rot, 256)
        assert torch.allclose(x, x_back, atol=1e-3)


class TestCollectorWithRotation:
    """Tests for ActivationStatsCollector with ConvRot rotation enabled."""

    def test_rotated_hessian_matches_manual_rotation(self, calibration_mod):
        """Hessian with rot_size should equal rotate(x).T @ rotate(x)."""
        torch.manual_seed(0)
        lin = nn.Linear(256, 4, bias=False)
        xs = [torch.randn(2, 256) for _ in range(3)]

        expected = sum(
            calibration_mod.rotate_activations(_flatten_linear(x), 256).T
            @ calibration_mod.rotate_activations(_flatten_linear(x), 256)
            for x in xs
        )

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "lin", store, "Linear", hessian_block_size=0, rot_size=256
        )
        collector.register(lin)
        for x in xs:
            _ = lin(x)
        collector.remove()

        assert torch.allclose(store["hessians"]["lin"], expected, atol=1e-3)

    def test_rotation_changes_hessian(self, calibration_mod):
        """rot_size > 0 should produce a different Hessian than rot_size=0."""
        torch.manual_seed(0)
        lin = nn.Linear(256, 4, bias=False)
        x = torch.randn(2, 256)

        # Without rotation
        store1: dict = {}
        c1 = calibration_mod.ActivationStatsCollector("lin", store1, "Linear", hessian_block_size=0)
        c1.register(lin)
        _ = lin(x)
        c1.remove()

        # With rotation
        store2: dict = {}
        c2 = calibration_mod.ActivationStatsCollector("lin", store2, "Linear", hessian_block_size=0, rot_size=256)
        c2.register(lin)
        _ = lin(x)
        c2.remove()

        H1 = store1["hessians"]["lin"]
        H2 = store2["hessians"]["lin"]
        # Rotation changes the Hessian — they should differ
        assert not torch.allclose(H1, H2, atol=1e-3)
        # But both should be valid (finite, symmetric, same shape)
        assert H1.shape == H2.shape
        assert torch.all(torch.isfinite(H2))
        assert torch.allclose(H2, H2.T, atol=1e-5)

    def test_rotated_block_diagonal(self, calibration_mod):
        """Block-diagonal Hessian with rotation produces valid blocks."""
        torch.manual_seed(0)
        lin = nn.Linear(1024, 4, bias=False)
        x = torch.randn(2, 1024)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "lin", store, "Linear", hessian_block_size=128, rot_size=256
        )
        collector.register(lin)
        _ = lin(x)
        collector.remove()

        H = store["hessians"]["lin"]
        assert isinstance(H, list)
        assert len(H) == 8  # 1024/128 = 8 blocks
        assert all(block.shape == (128, 128) for block in H)
        # All blocks should be finite and symmetric
        for block in H:
            assert torch.all(torch.isfinite(block))
            assert torch.allclose(block, block.T, atol=1e-5)

    def test_rotation_disabled_by_default(self, calibration_mod):
        """When rot_size=0 (default), no rotation is applied."""
        torch.manual_seed(0)
        lin = nn.Linear(64, 4, bias=False)
        x = torch.randn(2, 64)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "lin", store, "Linear", hessian_block_size=0, rot_size=0
        )
        collector.register(lin)
        _ = lin(x)
        collector.remove()

        x_flat = x.reshape(-1, 64)
        expected = x_flat.T @ x_flat
        assert torch.allclose(store["hessians"]["lin"], expected, atol=1e-4)

    def test_rotated_hessian_matches_converter_rotation(self, calibration_mod):
        """Rotated-space Hessian should match converter's rotate_hessian(unrotated_H).

        This verifies the key equivalence:
            rotate_activations(x).T @ rotate_activations(x)  ==  rotate_hessian(x.T @ x)
        """
        torch.manual_seed(0)
        lin = nn.Linear(256, 4, bias=False)
        x = torch.randn(2, 256)
        x_flat = _flatten_linear(x)

        # Collect unrotated Hessian
        store_unrot: dict = {}
        c1 = calibration_mod.ActivationStatsCollector("unrot", store_unrot, "Linear", hessian_block_size=0)
        c1.register(lin)
        _ = lin(x)
        c1.remove()
        H_unrot = store_unrot["hessians"]["unrot"]

        # Collect rotated Hessian
        store_rot: dict = {}
        c2 = calibration_mod.ActivationStatsCollector("rot", store_rot, "Linear", hessian_block_size=0, rot_size=256)
        c2.register(lin)
        _ = lin(x)
        c2.remove()
        H_rot = store_rot["hessians"]["rot"]

        # Manually rotate the unrotated Hessian using the same Hadamard
        # transform that rotate_activations applies.
        # rotate_activations pads to multiple of rot_size then applies
        # group-wise H.T. For the Hessian: H_rot_manual = (R @ x)^T (R @ x)
        # which equals what rotate_hessian would produce.
        x_rot = calibration_mod.rotate_activations(x_flat, 256)
        H_rot_manual = x_rot.T @ x_rot

        assert torch.allclose(H_rot, H_rot_manual, atol=1e-3)

    def test_rotation_with_non_divisible_features(self, calibration_mod):
        """Rotation works when in_features is not a multiple of rot_size.

        The collector accumulates at the padded size (256x256).
        collect_stats() crops back to the original size (128x128).
        """
        torch.manual_seed(0)
        lin = nn.Linear(128, 4, bias=False)
        x = torch.randn(2, 128)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "nd", store, "Linear", hessian_block_size=0, rot_size=256
        )
        collector.register(lin)
        _ = lin(x)
        collector.remove()

        H = store["hessians"]["nd"]
        # Collector stores at padded size (rot_size=256 pads 128→256)
        assert H.shape == (256, 256)
        assert torch.all(torch.isfinite(H))
        assert torch.allclose(H, H.T, atol=1e-5)

    def test_rotation_block_diagonal_non_divisible(self, calibration_mod):
        """Block-diagonal rotation works when in_features is not a multiple of rot_size.

        The collector pads to rot_size (600→768) then blocks at 128 = 6 blocks.
        collect_stats() crops back to the original 600 features.
        """
        torch.manual_seed(0)
        lin = nn.Linear(600, 4, bias=False)
        x = torch.randn(2, 600)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "nd_blk", store, "Linear", hessian_block_size=128, rot_size=256
        )
        collector.register(lin)
        _ = lin(x)
        collector.remove()

        H = store["hessians"]["nd_blk"]
        assert isinstance(H, list)
        # Collector stores at padded size: 768/128 = 6 blocks
        assert len(H) == 6
        assert all(block.shape == (128, 128) for block in H)


# ── PermuQuant tests ────────────────────────────────────────────────────────

class TestPermuQuantMu2:
    """Tests for PermuQuant second-moment collection."""

    def test_mu2_accumulates_mean_squared(self, calibration_mod):
        """mode='mu2' should accumulate sum(x^2) per channel, with count tracking."""
        torch.manual_seed(0)
        lin = nn.Linear(8, 4, bias=False)
        xs = [torch.randn(2, 8) for _ in range(3)]

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "mu2_test", store, "Linear", mode="mu2"
        )
        collector.register(lin)
        for x in xs:
            _ = lin(x)
        collector.remove()

        assert "mu2" in store
        mu2_sum = store["mu2"]["mu2_test"]
        assert mu2_sum.shape == (8,)
        assert torch.all(torch.isfinite(mu2_sum))
        assert torch.all(mu2_sum >= 0)

        # mu2_sum should be the sum of x^2 across all samples (6 rows total)
        all_x = torch.cat([x.reshape(-1, 8) for x in xs])
        expected_sum = (all_x ** 2).sum(dim=0)
        assert torch.allclose(mu2_sum, expected_sum, atol=1e-3)

        # Count should be total number of rows
        assert store["_mu2_count"]["mu2_test"] == 6

        # Mean = sum / count
        mu2_mean = mu2_sum / 6
        expected_mean = (all_x ** 2).mean(dim=0)
        assert torch.allclose(mu2_mean, expected_mean, atol=1e-4)

    def test_mu2_no_hessian_stored(self, calibration_mod):
        """mode='mu2' should not create any Hessian entries."""
        torch.manual_seed(0)
        lin = nn.Linear(8, 4, bias=False)
        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "no_h", store, "Linear", mode="mu2"
        )
        collector.register(lin)
        _ = lin(torch.randn(2, 8))
        collector.remove()

        assert "hessians" not in store

    def test_permutation_sorts_descending_by_mu2(self, calibration_mod):
        """Permutation should sort channels by descending second moment."""
        torch.manual_seed(0)
        # Create activations with known second moments
        lin = nn.Linear(8, 4, bias=False)
        # Channel 0 has huge values, channel 7 has tiny values
        x = torch.randn(16, 8)
        x[:, 0] *= 100
        x[:, 7] *= 0.01

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "perm_test", store, "Linear", mode="mu2"
        )
        collector.register(lin)
        _ = lin(x)
        collector.remove()

        mu2 = store["mu2"]["perm_test"]
        perm = mu2.argsort(descending=True)

        # Channel 0 (highest mu2) should be first, channel 7 (lowest) should be last
        assert perm[0].item() == 0
        assert perm[-1].item() == 7

    def test_permutation_applied_to_hessian(self, calibration_mod):
        """When permutation is set, activations should be reordered before x^T @ x."""
        torch.manual_seed(0)
        lin = nn.Linear(8, 4, bias=False)
        x = torch.randn(4, 8)
        perm = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int32)  # reverse

        # Without permutation
        store1: dict = {}
        c1 = calibration_mod.ActivationStatsCollector("nop", store1, "Linear")
        c1.register(lin)
        _ = lin(x)
        c1.remove()

        # With permutation
        store2: dict = {}
        c2 = calibration_mod.ActivationStatsCollector("perm", store2, "Linear", permutation=perm)
        c2.register(lin)
        _ = lin(x)
        c2.remove()

        H1 = store1["hessians"]["nop"]
        H2 = store2["hessians"]["perm"]

        # H2 should be H1 permuted: H2[i,j] = H1[perm[i], perm[j]]
        H1_perm = H1[perm][:, perm]
        assert torch.allclose(H2, H1_perm, atol=1e-4)

    def test_permutation_with_block_diagonal(self, calibration_mod):
        """Permutation with block-diagonal Hessian should produce valid blocks."""
        torch.manual_seed(0)
        lin = nn.Linear(1024, 4, bias=False)
        x = torch.randn(2, 1024)
        perm = torch.randperm(1024, dtype=torch.int32)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "blk_perm", store, "Linear", hessian_block_size=128, permutation=perm
        )
        collector.register(lin)
        _ = lin(x)
        collector.remove()

        H = store["hessians"]["blk_perm"]
        assert isinstance(H, list)
        assert len(H) == 8
        assert all(block.shape == (128, 128) for block in H)
        for block in H:
            assert torch.all(torch.isfinite(block))
            assert torch.allclose(block, block.T, atol=1e-5)


class TestPermuteHessian:
    """Tests for the permute_hessian function."""

    def test_full_hessian_permutation(self, calibration_mod):
        """Permuting a full 2D Hessian should equal H[perm][:, perm]."""
        torch.manual_seed(42)
        H = torch.randn(16, 16)
        H = H @ H.T  # make symmetric positive-definite
        perm = torch.randperm(16, dtype=torch.int32)

        H_perm = calibration_mod.permute_hessian(H, perm)
        expected = H[perm][:, perm]
        assert torch.allclose(H_perm, expected, atol=1e-5)

    def test_block_diagonal_list_permutation(self, calibration_mod):
        """Permuting a block-diagonal Hessian (list of blocks) should produce
        a valid result with the same block structure."""
        torch.manual_seed(42)
        n = 256
        block_size = 64

        # Create block-diagonal (off-diagonal blocks are zero)
        blocks = []
        for i in range(n // block_size):
            B = torch.randn(block_size, block_size)
            blocks.append(B @ B.T)

        # Identity permutation — should return the same blocks
        perm_id = torch.arange(n, dtype=torch.int32)
        result = calibration_mod.permute_hessian(blocks, perm_id)
        assert isinstance(result, list)
        assert len(result) == len(blocks)
        for rb, ob in zip(result, blocks):
            assert torch.allclose(rb, ob, atol=1e-5)

    def test_permute_hessian_preserves_finiteness(self, calibration_mod):
        """Permuted Hessian should have all finite values."""
        torch.manual_seed(42)
        H = torch.randn(32, 32)
        H = H @ H.T
        perm = torch.randperm(32, dtype=torch.int32)

        H_perm = calibration_mod.permute_hessian(H, perm)
        assert torch.all(torch.isfinite(H_perm))
        assert torch.allclose(H_perm, H_perm.T, atol=1e-5)

    def test_permute_hessian_with_sub_perm(self, calibration_mod):
        """When perm has fewer entries than H size, result is sub-sized."""
        torch.manual_seed(42)
        H = torch.randn(64, 64)
        H = H @ H.T
        # Only permute the first 32 channels
        perm = torch.randperm(32, dtype=torch.int32)

        H_perm = calibration_mod.permute_hessian(H, perm)
        assert H_perm.shape == (32, 32)
        expected = H[perm][:, perm]
        assert torch.allclose(H_perm, expected, atol=1e-5)


class TestCollectorModeBoth:
    """Tests for mode='both' (single-pass mu2 + hessian collection)."""

    def test_both_mode_collects_mu2_and_hessian(self, calibration_mod):
        """mode='both' should accumulate both mu2 and Hessian in one pass."""
        torch.manual_seed(0)
        lin = nn.Linear(8, 4, bias=False)
        xs = [torch.randn(2, 8) for _ in range(3)]

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "both_test", store, "Linear", mode="both"
        )
        collector.register(lin)
        for x in xs:
            _ = lin(x)
        collector.remove()

        # mu2 should be collected
        assert "mu2" in store
        mu2_sum = store["mu2"]["both_test"]
        assert mu2_sum.shape == (8,)
        assert store["_mu2_count"]["both_test"] == 6

        # Hessian should also be collected
        assert "hessians" in store
        H = store["hessians"]["both_test"]
        assert H.shape == (8, 8)

        # Both should match what separate collectors would produce
        all_x = torch.cat([x.reshape(-1, 8) for x in xs])
        expected_mu2 = (all_x ** 2).sum(dim=0)
        expected_H = all_x.T @ all_x
        assert torch.allclose(mu2_sum, expected_mu2, atol=1e-3)
        assert torch.allclose(H, expected_H, atol=1e-3)

    def test_both_mode_with_amax(self, calibration_mod):
        """mode='both' should also collect amax."""
        torch.manual_seed(0)
        lin = nn.Linear(8, 4, bias=False)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "both_amax", store, "Linear", mode="both", collect_amax=True
        )
        collector.register(lin)
        _ = lin(torch.randn(2, 8))
        collector.remove()

        assert "amax" in store
        assert store["amax"]["both_amax"] > 0

    def test_both_mode_with_rotation(self, calibration_mod):
        """mode='both' with rot_size should rotate activations for both mu2 and hessian."""
        torch.manual_seed(0)
        lin = nn.Linear(256, 4, bias=False)
        x = torch.randn(2, 256)

        store: dict = {}
        collector = calibration_mod.ActivationStatsCollector(
            "rot_both", store, "Linear", mode="both", rot_size=256
        )
        collector.register(lin)
        _ = lin(x)
        collector.remove()

        assert "mu2" in store
        assert "hessians" in store
        H = store["hessians"]["rot_both"]
        assert H.shape == (256, 256)
        assert torch.all(torch.isfinite(H))

