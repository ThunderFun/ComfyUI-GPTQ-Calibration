"""Tests for Triton FHT rotation kernels.

Validates that the Triton FHT kernels produce the same result as the
reference ``rotate_activations()`` from calibration.py for all supported
rot_sizes and input shapes.

Run with:
    pytest tests/test_fht_rotate.py -v
    pytest tests/test_fht_rotate.py -v -k "cuda"   # GPU-only tests
    pytest tests/test_fht_rotate.py -v -k "cpu"     # CPU-only tests
"""

import math
import pytest
import torch
import torch.nn.functional as F

# ── Reference implementation (duplicated from calibration.py) ──────────────────
# Duplicated rather than imported to avoid pulling in the ComfyUI stubs
# that conftest.py installs — keeps these tests runnable in isolation.

_HADAMARD_CACHE: dict = {}


def _is_power_of_four(n: int) -> bool:
    if n < 4:
        return False
    return (n & (n - 1)) == 0 and (n & 0x55555555) == n


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def get_hadamard(size: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Reference Hadamard matrix construction (Regular for power-of-4, Sylvester otherwise)."""
    key = (size, str(dtype), device)
    if key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[key]
    if not ((size & (size - 1)) == 0 and size > 0):
        raise ValueError(f"Hadamard size must be a power of 2, got {size}")

    if _is_power_of_four(size):
        H4 = torch.tensor([
            [ 1.0,  1.0,  1.0, -1.0],
            [ 1.0,  1.0, -1.0,  1.0],
            [ 1.0, -1.0,  1.0,  1.0],
            [-1.0,  1.0,  1.0,  1.0],
        ], dtype=dtype, device=device) / 2.0
        H = H4
        while H.shape[0] < size:
            H = torch.kron(H, H4)
    else:
        H = torch.tensor([[1.0]], dtype=dtype, device=device)
        while H.shape[0] < size:
            H = torch.kron(
                torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device),
                H,
            )
        H = H * (1.0 / math.sqrt(size))

    _HADAMARD_CACHE[key] = H
    return H


def rotate_activations_ref(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Reference group-wise Hadamard rotation (from calibration.py)."""
    if not _is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be a power of 2, got {rot_size}")
    orig_features = x.shape[-1]
    if orig_features % rot_size != 0:
        pad = rot_size - (orig_features % rot_size)
        x = F.pad(x, (0, pad))
    in_features = x.shape[-1]
    num_groups = in_features // rot_size
    leading_shape = x.shape[:-1]
    x_flat = x.reshape(-1, num_groups, rot_size)
    H = get_hadamard(rot_size, x.dtype, x.device)
    x_rotated = torch.matmul(x_flat, H.T)
    return x_rotated.reshape(*leading_shape, in_features)


# ── Import kernel under test ────────────────────────────────────────────────

# The kernel module lives at kernels/triton_fht_rotate.py
# We import it here so the test fails loudly if the module doesn't exist yet.
try:
    from kernels.triton_fht_rotate import fht_rotate, _HAS_TRITON
except ImportError:
    # Allow running CPU-only tests even if the kernel module isn't built yet.
    fht_rotate = None
    _HAS_TRITON = False


# ── Fixtures ────────────────────────────────────────────────────────────────

HAS_CUDA = torch.cuda.is_available()


def _skip_if_no_cuda():
    if not HAS_CUDA:
        pytest.skip("CUDA not available")


def _skip_if_no_triton():
    if not _HAS_TRITON:
        pytest.skip("Triton not available")


# ── Regular Hadamard FHT (power-of-4) ──────────────────────────────────────

class TestRegularHadamardFHT:
    """Tests for power-of-4 rot_sizes: 16, 64, 256, 1024."""

    @pytest.mark.parametrize("rot_size", [16, 64, 256])
    @pytest.mark.parametrize("M,K", [
        (1, 256),       # single row, minimal
        (4, 512),       # small batch
        (32, 2048),     # typical layer
        (128, 4096),    # larger layer
    ])
    def test_matches_reference_cuda(self, rot_size, M, K):
        """Triton FHT output must match reference rotate_activations on CUDA."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        if K % rot_size != 0:
            pytest.skip(f"K={K} not divisible by rot_size={rot_size}")

        x = torch.randn(M, K, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)

        # FHT butterfly stages accumulate rounding from the 4-element
        # butterfly pattern.  Tolerance of 1e-4 relative to the reference
        # matmul is acceptable (the reference itself is exact to float32).
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-4, (
            f"rot_size={rot_size}, M={M}, K={K}: "
            f"max_err={max_err:.6f}, rel_err={rel_err:.2e}"
        )

    @pytest.mark.parametrize("rot_size", [16, 64, 256])
    def test_matches_reference_cpu_fallback(self, rot_size):
        """CPU fallback must produce identical results to reference."""
        M, K = 16, 512
        x = torch.randn(M, K, dtype=torch.float32)
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)  # should fall back to CPU matmul
        assert torch.allclose(ref, out, atol=1e-6), (
            f"CPU fallback mismatch for rot_size={rot_size}"
        )


# ── Sylvester Hadamard FHT (power-of-2, non-power-of-4) ───────────────────

class TestSylvesterFHT:
    """Tests for non-power-of-4 rot_sizes: 8, 32, 128, 512."""

    @pytest.mark.parametrize("rot_size", [8, 32, 128])
    @pytest.mark.parametrize("M,K", [
        (1, 128),
        (4, 256),
        (32, 2048),
        (64, 4096),
    ])
    def test_matches_reference_cuda(self, rot_size, M, K):
        """Triton Sylvester FHT must match reference on CUDA."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        if K % rot_size != 0:
            pytest.skip(f"K={K} not divisible by rot_size={rot_size}")

        x = torch.randn(M, K, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)

        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-4, (
            f"rot_size={rot_size}, M={M}, K={K}: "
            f"max_err={max_err:.6f}, rel_err={rel_err:.2e}"
        )

    @pytest.mark.parametrize("rot_size", [8, 32, 128])
    def test_matches_reference_cpu_fallback(self, rot_size):
        """CPU fallback must produce identical results to reference."""
        M, K = 16, 512
        x = torch.randn(M, K, dtype=torch.float32)
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)
        assert torch.allclose(ref, out, atol=1e-6), (
            f"CPU fallback mismatch for rot_size={rot_size}"
        )

    def test_rot_size_512_cuda(self):
        """rot_size=512 (9 Sylvester stages) — real use case."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        M, K = 16, 2048
        x = torch.randn(M, K, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, 512)
        out = fht_rotate(x, 512)
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-3


# ── Padding (K not divisible by rot_size) ──────────────────────────────────

class TestPadding:
    """When K is not a multiple of rot_size, both reference and kernel pad
    internally and return the padded shape (K_padded)."""

    @pytest.mark.parametrize("rot_size,K_in", [
        (256, 300),     # 300 → pad to 512
        (64, 100),      # 100 → pad to 128
        (16, 50),       # 50  → pad to 64
        (128, 200),     # 200 → pad to 256
    ])
    def test_padded_input_cuda(self, rot_size, K_in):
        """Output must match reference when K is not a multiple of rot_size.

        Both the reference and the FHT kernel pad internally and return
        the padded shape (K_padded).  The caller (calibration hook) crops
        to original features after Hessian accumulation.
        """
        _skip_if_no_cuda()
        _skip_if_no_triton()
        M = 8
        K_padded = math.ceil(K_in / rot_size) * rot_size
        x = torch.randn(M, K_in, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)
        assert out.shape == ref.shape == (M, K_padded), (
            f"Shape mismatch: got {out.shape}, ref {ref.shape}, expected ({M}, {K_padded})"
        )
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("rot_size,K_in", [
        (256, 300),
        (64, 100),
    ])
    def test_padded_input_cpu(self, rot_size, K_in):
        """CPU fallback with padding must match reference."""
        M = 8
        K_padded = math.ceil(K_in / rot_size) * rot_size
        x = torch.randn(M, K_in, dtype=torch.float32)
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)
        assert out.shape == (M, K_padded)
        assert torch.allclose(ref, out, atol=1e-6)


# ── 3D input tensors (batch dimension) ─────────────────────────────────────

class TestBatchedInput:
    """The kernel should handle 3D inputs [B, S, K] by flattening to [B*S, K]."""

    @pytest.mark.parametrize("rot_size", [64, 256])
    def test_3d_input_cuda(self, rot_size):
        """3D [B, S, K] input must be flattened, rotated, then reshaped back."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        B, S, K = 2, 16, 1024
        x = torch.randn(B, S, K, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)
        assert out.shape == (B, S, K)
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("rot_size", [64, 256])
    def test_3d_input_cpu(self, rot_size):
        """CPU fallback must also handle 3D inputs correctly."""
        B, S, K = 2, 16, 1024
        x = torch.randn(B, S, K, dtype=torch.float32)
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)
        assert out.shape == (B, S, K)
        assert torch.allclose(ref, out, atol=1e-6)


# ── Hadamard transform properties ──────────────────────────────────────────

class TestTransformProperties:
    """The FHT output should preserve key mathematical properties."""

    @pytest.mark.parametrize("rot_size", [16, 64, 256, 128])
    def test_orthogonality_preserves_norm(self, rot_size):
        """‖Hx‖₂ should equal ‖x‖₂ (Hadamard is orthogonal up to normalization)."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        M, K = 16, rot_size * 4
        x = torch.randn(M, K, dtype=torch.float32, device="cuda")
        x_rot = fht_rotate(x, rot_size)
        # Norms should be preserved within each group
        x_groups = x.reshape(M, -1, rot_size)
        x_rot_groups = x_rot.reshape(M, -1, rot_size)
        norm_in = x_groups.norm(dim=2)
        norm_out = x_rot_groups.norm(dim=2)
        assert torch.allclose(norm_in, norm_out, atol=1e-3, rtol=1e-3), (
            f"Norm not preserved for rot_size={rot_size}: "
            f"max diff = {(norm_in - norm_out).abs().max().item():.6f}"
        )

    @pytest.mark.parametrize("rot_size", [16, 64, 256])
    def test_inverse_transform_recovers_input(self, rot_size):
        """H @ H.T = I, so applying the transform twice should recover input."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        M, K = 8, rot_size * 2
        x = torch.randn(M, K, dtype=torch.float32, device="cuda")
        x_rot = fht_rotate(x, rot_size)
        x_inv = fht_rotate(x_rot, rot_size)
        assert torch.allclose(x, x_inv, atol=1e-3), (
            f"Double transform does not recover input for rot_size={rot_size}: "
            f"max diff = {(x - x_inv).abs().max().item():.6f}"
        )


# ── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_row(self):
        """M=1 should work."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        x = torch.randn(1, 256, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, 64)
        out = fht_rotate(x, 64)
        assert out.shape == (1, 256)
        assert (ref - out).abs().max().item() < 1e-4

    def test_rot_size_equals_K(self):
        """When rot_size == K, there's exactly one group."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        x = torch.randn(8, 256, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, 256)
        out = fht_rotate(x, 256)
        assert torch.allclose(ref, out, atol=1e-4)

    def test_large_rot_size(self):
        """rot_size=1024 should work (5 stages)."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        x = torch.randn(4, 2048, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, 1024)
        out = fht_rotate(x, 1024)
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-3

    def test_rot_size_4096(self):
        """rot_size=4096 should work (6 stages, max for Regular Hadamard kernel)."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        x = torch.randn(4, 4096, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, 4096)
        out = fht_rotate(x, 4096)
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-3

    def test_invalid_rot_size_raises(self):
        """Non-power-of-2 rot_size should raise ValueError."""
        x = torch.randn(4, 256, dtype=torch.float32)
        with pytest.raises(ValueError):
            fht_rotate(x, 100)

    def test_empty_tensor(self):
        """M=0 should return empty tensor."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        x = torch.randn(0, 256, dtype=torch.float32, device="cuda")
        out = fht_rotate(x, 64)
        assert out.shape == (0, 256)


# ── dtype handling (bf16/fp16 inputs from model) ───────────────────────────

class TestDtypeHandling:
    """The calibration hook receives fp16/bf16 activations from the model,
    casts to float32, then rotates.  Verify the cast+FHT path matches
    the cast+matmul reference."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("rot_size", [64, 256])
    def test_half_precision_input_cuda(self, dtype, rot_size):
        _skip_if_no_cuda()
        _skip_if_no_triton()
        M, K = 32, 1024
        x = torch.randn(M, K, dtype=dtype, device="cuda")
        # Reference: cast to float32, rotate, cast back to half
        ref = rotate_activations_ref(x.float(), rot_size).to(dtype)
        # Kernel: fht_rotate casts internally and casts back to orig dtype
        out = fht_rotate(x, rot_size)
        assert out.dtype == dtype, f"Output dtype {out.dtype} != input {dtype}"
        # Compare in the original dtype — both sides went through the
        # same float32→half round-trip, so this tests FHT correctness
        # without conflating it with dtype precision loss.
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.float().abs().max().item() + 1e-8)
        assert rel_err < 1e-4, (
            f"dtype={dtype}, rot_size={rot_size}: rel_err={rel_err:.2e}"
        )


# ── Non-contiguous inputs ──────────────────────────────────────────────────

class TestNonContiguous:
    """nn.Unfold and tensor slicing can produce non-contiguous tensors.
    The kernel must handle these correctly."""

    def test_transposed_input_cuda(self):
        """A transposed tensor is non-contiguous."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        M, K = 32, 1024
        x = torch.randn(K, M, dtype=torch.float32, device="cuda").T  # [M, K], non-contiguous
        assert not x.is_contiguous()
        ref = rotate_activations_ref(x, 256)
        out = fht_rotate(x, 256)
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-4

    def test_sliced_input_cuda(self):
        """Slicing produces a non-contiguous view."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        M, K = 32, 2048
        x_full = torch.randn(M, K, dtype=torch.float32, device="cuda")
        x = x_full[:, ::2]  # [M, 1024], non-contiguous stride
        assert not x.is_contiguous()
        ref = rotate_activations_ref(x.contiguous(), 256)
        out = fht_rotate(x, 256)
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-4


# ── Chunking (M > _CHUNK_ROWS) ────────────────────────────────────────────

class TestChunking:
    """The kernel processes rows in chunks of _CHUNK_ROWS=2048.
    Verify correctness when M exceeds this boundary."""

    @pytest.mark.parametrize("M", [2049, 4096, 5000])
    def test_large_M_cuda(self, M):
        """M > _CHUNK_ROWS must produce correct results across chunk boundaries."""
        _skip_if_no_cuda()
        _skip_if_no_triton()
        K, rot_size = 1024, 256
        x = torch.randn(M, K, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-4, (
            f"M={M}: rel_err={rel_err:.2e}"
        )


# ── rot_size > K (small layers) ───────────────────────────────────────────

class TestSmallLayers:
    """When rot_size > K, the padding logic must pad K up to rot_size.
    This happens with small Conv2d layers (e.g., 3×3 kernel, few channels)."""

    @pytest.mark.parametrize("rot_size,K_in", [
        (256, 64),    # 64 → pad to 256
        (64, 32),     # 32 → pad to 64
        (128, 100),   # 100 → pad to 128
    ])
    def test_rot_size_larger_than_K_cuda(self, rot_size, K_in):
        _skip_if_no_cuda()
        _skip_if_no_triton()
        M = 16
        K_padded = rot_size  # ceil(K_in / rot_size) * rot_size = rot_size
        x = torch.randn(M, K_in, dtype=torch.float32, device="cuda")
        ref = rotate_activations_ref(x, rot_size)
        out = fht_rotate(x, rot_size)
        assert out.shape == ref.shape == (M, K_padded)
        max_err = (ref - out).abs().max().item()
        rel_err = max_err / (ref.abs().max().item() + 1e-8)
        assert rel_err < 1e-4
