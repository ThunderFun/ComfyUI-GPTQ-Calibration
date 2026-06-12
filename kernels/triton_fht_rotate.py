"""FHT rotation kernels — O(N log N) Fast Hadamard Transform.

Drop-in replacement for the O(N²) matmul-based rotate_activations() in
calibration.py.  Two variants: Regular (power-of-4) and Sylvester (any power-of-2).
"""

import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ── Constants ───────────────────────────────────────────────────────────────

# Max rows per kernel launch — keeps tmp buffer memory bounded.
# With BLOCK_K=256: 2048 × 256 × 4 × 2 = 4 MiB per chunk.
_CHUNK_ROWS = 2048

# 1/sqrt(2) as a constant for the Sylvester butterfly.
_INV_SQRT2 = 0.7071067811865476


# ── Regular Hadamard FHT kernel (power-of-4) ───────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _fht_rotate_regular_kernel(
        x_ptr, out_ptr, tmp_a_ptr, tmp_b_ptr,
        stride_xm, stride_outm, stride_tmpm,
        K: tl.constexpr, rot_size: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
    ):
        """Group-wise Regular Hadamard FHT — power-of-4 rot_sizes.

        Each program processes one row divided into NUM_CHUNKS groups of
        BLOCK_K features.  Tmp buffers use local indices so each chunk
        reuses the same scratch space.
        """
        pid = tl.program_id(axis=0)
        x_base = x_ptr + pid * stride_xm
        out_base = out_ptr + pid * stride_outm
        tmp_a_row = tmp_a_ptr + pid * stride_tmpm
        tmp_b_row = tmp_b_ptr + pid * stride_tmpm

        for chunk in range(NUM_CHUNKS):
            g: tl.constexpr = chunk * BLOCK_K
            offs = tl.arange(0, BLOCK_K)
            cols = g + offs
            mask = cols < K

            # Load chunk from x (global index), store to tmp (local index)
            vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(tmp_a_row + offs, vals, mask=mask)
            tl.debug_barrier()

            # Stage 0 (s=1): tmp_a → tmp_b
            sub = (offs // 1) % 4
            base = offs - sub
            a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
            b = tl.load(tmp_a_row + base + 1, mask=mask, other=0.0)
            c = tl.load(tmp_a_row + base + 2, mask=mask, other=0.0)
            d = tl.load(tmp_a_row + base + 3, mask=mask, other=0.0)
            r0 = (a + b + c - d) * 0.5
            r1 = (a + b - c + d) * 0.5
            r2 = (a - b + c + d) * 0.5
            r3 = (-a + b + c + d) * 0.5
            result = tl.where(sub == 0, r0,
                     tl.where(sub == 1, r1,
                     tl.where(sub == 2, r2, r3)))
            tl.store(tmp_b_row + offs, result, mask=mask)
            tl.debug_barrier()

            if NUM_STAGES > 1:
                # Stage 1 (s=4): tmp_b → tmp_a
                sub = (offs // 4) % 4
                base = offs - sub * 4
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 4, mask=mask, other=0.0)
                c = tl.load(tmp_b_row + base + 8, mask=mask, other=0.0)
                d = tl.load(tmp_b_row + base + 12, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 2:
                # Stage 2 (s=16): tmp_a → tmp_b
                sub = (offs // 16) % 4
                base = offs - sub * 16
                a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_a_row + base + 16, mask=mask, other=0.0)
                c = tl.load(tmp_a_row + base + 32, mask=mask, other=0.0)
                d = tl.load(tmp_a_row + base + 48, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_b_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 3:
                # Stage 3 (s=64): tmp_b → tmp_a
                sub = (offs // 64) % 4
                base = offs - sub * 64
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 64, mask=mask, other=0.0)
                c = tl.load(tmp_b_row + base + 128, mask=mask, other=0.0)
                d = tl.load(tmp_b_row + base + 192, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 4:
                # Stage 4 (s=256): tmp_a → tmp_b
                sub = (offs // 256) % 4
                base = offs - sub * 256
                a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_a_row + base + 256, mask=mask, other=0.0)
                c = tl.load(tmp_a_row + base + 512, mask=mask, other=0.0)
                d = tl.load(tmp_a_row + base + 768, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_b_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 5:
                # Stage 5 (s=1024): tmp_b → tmp_a
                sub = (offs // 1024) % 4
                base = offs - sub * 1024
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 1024, mask=mask, other=0.0)
                c = tl.load(tmp_b_row + base + 2048, mask=mask, other=0.0)
                d = tl.load(tmp_b_row + base + 3072, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            # Final result is in tmp_a if NUM_STAGES is even, tmp_b if odd
            if NUM_STAGES % 2 == 0:
                final = tl.load(tmp_a_row + offs, mask=mask, other=0.0)
            else:
                final = tl.load(tmp_b_row + offs, mask=mask, other=0.0)

            # Store to output using global index
            tl.store(out_base + cols, final, mask=mask)


# ── Sylvester Hadamard FHT kernel (any power-of-2) ─────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _fht_rotate_sylvester_kernel(
        x_ptr, out_ptr, tmp_a_ptr, tmp_b_ptr,
        stride_xm, stride_outm, stride_tmpm,
        K: tl.constexpr, rot_size: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
    ):
        """Group-wise Sylvester Hadamard FHT — any power-of-2 rot_size.

        Each program processes one row divided into NUM_CHUNKS groups of
        BLOCK_K features.  Uses the 2-element butterfly: (a ± b) / √2.
        """
        pid = tl.program_id(axis=0)
        x_base = x_ptr + pid * stride_xm
        out_base = out_ptr + pid * stride_outm
        tmp_a_row = tmp_a_ptr + pid * stride_tmpm
        tmp_b_row = tmp_b_ptr + pid * stride_tmpm

        inv_sqrt2: tl.constexpr = 0.7071067811865476

        for chunk in range(NUM_CHUNKS):
            g: tl.constexpr = chunk * BLOCK_K
            offs = tl.arange(0, BLOCK_K)
            cols = g + offs
            mask = cols < K

            # Load chunk from x (global index), store to tmp (local index)
            vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(tmp_a_row + offs, vals, mask=mask)
            tl.debug_barrier()

            # Apply NUM_STAGES butterfly stages, alternating tmp_a ↔ tmp_b.
            # Stage 0 (stride=1): pairs (0,1), (2,3), ...
            # Stage 1 (stride=2): pairs (0,2), (1,3), (4,6), ...
            # Stage 2 (stride=4): pairs (0,4), (1,5), (2,6), ...
            # ...
            # Stage i (stride=2^i): pairs within groups of 2^{i+1}

            if NUM_STAGES > 0:
                # Stage 0 (stride=1): tmp_a → tmp_b
                sub = (offs // 1) % 2
                base = offs - sub
                a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_a_row + base + 1, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_b_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 1:
                # Stage 1 (stride=2): tmp_b → tmp_a
                sub = (offs // 2) % 2
                base = offs - sub * 2
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 2, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 2:
                # Stage 2 (stride=4): tmp_a → tmp_b
                sub = (offs // 4) % 2
                base = offs - sub * 4
                a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_a_row + base + 4, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_b_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 3:
                # Stage 3 (stride=8): tmp_b → tmp_a
                sub = (offs // 8) % 2
                base = offs - sub * 8
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 8, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 4:
                # Stage 4 (stride=16): tmp_a → tmp_b
                sub = (offs // 16) % 2
                base = offs - sub * 16
                a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_a_row + base + 16, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_b_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 5:
                # Stage 5 (stride=32): tmp_b → tmp_a
                sub = (offs // 32) % 2
                base = offs - sub * 32
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 32, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 6:
                # Stage 6 (stride=64): tmp_a → tmp_b
                sub = (offs // 64) % 2
                base = offs - sub * 64
                a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_a_row + base + 64, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_b_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 7:
                # Stage 7 (stride=128): tmp_b → tmp_a
                sub = (offs // 128) % 2
                base = offs - sub * 128
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 128, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 8:
                # Stage 8 (stride=256): tmp_a → tmp_b
                sub = (offs // 256) % 2
                base = offs - sub * 256
                a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_a_row + base + 256, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_b_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 9:
                # Stage 9 (stride=512): tmp_b → tmp_a
                sub = (offs // 512) % 2
                base = offs - sub * 512
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 512, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 10:
                # Stage 10 (stride=1024): tmp_a → tmp_b
                sub = (offs // 1024) % 2
                base = offs - sub * 1024
                a = tl.load(tmp_a_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_a_row + base + 1024, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_b_row + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 11:
                # Stage 11 (stride=2048): tmp_b → tmp_a
                sub = (offs // 2048) % 2
                base = offs - sub * 2048
                a = tl.load(tmp_b_row + base, mask=mask, other=0.0)
                b = tl.load(tmp_b_row + base + 2048, mask=mask, other=0.0)
                r0 = (a + b) * inv_sqrt2
                r1 = (a - b) * inv_sqrt2
                result = tl.where(sub == 0, r0, r1)
                tl.store(tmp_a_row + offs, result, mask=mask)
                tl.debug_barrier()

            # Final result is in tmp_a if NUM_STAGES is even, tmp_b if odd
            if NUM_STAGES % 2 == 0:
                final = tl.load(tmp_a_row + offs, mask=mask, other=0.0)
            else:
                final = tl.load(tmp_b_row + offs, mask=mask, other=0.0)

            # Store to output using global index
            tl.store(out_base + cols, final, mask=mask)


# ── Helper functions ────────────────────────────────────────────────────────

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _is_power_of_four(n: int) -> bool:
    if n < 4:
        return False
    return (n & (n - 1)) == 0 and (n & 0x55555555) == n


# ── Reference implementation (CPU fallback) ────────────────────────────────

_HADAMARD_CACHE: dict = {}


def _get_hadamard(size: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Return a normalized Hadamard matrix (Regular for power-of-4, Sylvester otherwise)."""
    key = (size, str(dtype), device)
    if key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[key]

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


def _rotate_activations_cpu(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """CPU fallback: group-wise Hadamard rotation via dense matmul."""
    orig_features = x.shape[-1]
    if orig_features % rot_size != 0:
        pad = rot_size - (orig_features % rot_size)
        x = F.pad(x, (0, pad))
    in_features = x.shape[-1]
    num_groups = in_features // rot_size
    leading_shape = x.shape[:-1]
    x_flat = x.reshape(-1, num_groups, rot_size)
    H = _get_hadamard(rot_size, x.dtype, x.device)
    x_rotated = torch.matmul(x_flat, H.T)
    return x_rotated.reshape(*leading_shape, in_features)


# ── Triton dispatchers ─────────────────────────────────────────────────────

def _fht_rotate_cuda_regular(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Triton FHT for power-of-4 rot_sizes. Returns padded tensor; caller crops."""
    orig_features = x.shape[-1]
    if orig_features % rot_size != 0:
        pad = rot_size - (orig_features % rot_size)
        x = F.pad(x, (0, pad))

    M, K = x.shape
    if M == 0:
        return torch.empty(0, K, dtype=torch.float32, device=x.device)

    x_contig = x if x.is_contiguous() else x.contiguous()
    num_stages = int(math.log(rot_size) / math.log(4))
    num_chunks = K // rot_size
    block_k = rot_size  # must be power of 2

    out = torch.empty(M, K, dtype=torch.float32, device=x.device)

    # Allocate tmp buffers for one row chunk (reused across all row chunks)
    chunk = min(_CHUNK_ROWS, M)
    tmp_a = torch.empty(chunk, block_k, dtype=torch.float32, device=x.device)
    tmp_b = torch.empty(chunk, block_k, dtype=torch.float32, device=x.device)

    for start in range(0, M, _CHUNK_ROWS):
        end = min(start + _CHUNK_ROWS, M)
        n_rows = end - start

        x_chunk = x_contig[start:end]
        out_chunk = out[start:end]

        _fht_rotate_regular_kernel[(n_rows,)](
            x_chunk, out_chunk,
            tmp_a[:n_rows], tmp_b[:n_rows],
            x_chunk.stride(0), out_chunk.stride(0), tmp_a.stride(0),
            K, rot_size,
            NUM_STAGES=num_stages,
            BLOCK_K=block_k,
            NUM_CHUNKS=num_chunks,
        )

    return out


def _fht_rotate_cuda_sylvester(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Triton FHT for any power-of-2 rot_size. Returns padded tensor; caller crops."""
    orig_features = x.shape[-1]
    if orig_features % rot_size != 0:
        pad = rot_size - (orig_features % rot_size)
        x = F.pad(x, (0, pad))

    M, K = x.shape
    if M == 0:
        return torch.empty(0, K, dtype=torch.float32, device=x.device)

    x_contig = x if x.is_contiguous() else x.contiguous()
    num_stages = int(math.log(rot_size) / math.log(2))
    num_chunks = K // rot_size
    block_k = rot_size  # must be power of 2

    out = torch.empty(M, K, dtype=torch.float32, device=x.device)

    chunk = min(_CHUNK_ROWS, M)
    tmp_a = torch.empty(chunk, block_k, dtype=torch.float32, device=x.device)
    tmp_b = torch.empty(chunk, block_k, dtype=torch.float32, device=x.device)

    for start in range(0, M, _CHUNK_ROWS):
        end = min(start + _CHUNK_ROWS, M)
        n_rows = end - start

        x_chunk = x_contig[start:end]
        out_chunk = out[start:end]

        _fht_rotate_sylvester_kernel[(n_rows,)](
            x_chunk, out_chunk,
            tmp_a[:n_rows], tmp_b[:n_rows],
            x_chunk.stride(0), out_chunk.stride(0), tmp_a.stride(0),
            K, rot_size,
            NUM_STAGES=num_stages,
            BLOCK_K=block_k,
            NUM_CHUNKS=num_chunks,
        )

    return out


# ── Public API ──────────────────────────────────────────────────────────────

def fht_rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply group-wise Hadamard rotation using O(N log N) FHT.

    Drop-in for ``rotate_activations(x, rot_size)``.  Selects Regular vs
    Sylvester kernel based on ``rot_size``, with automatic CPU fallback.
    """
    if not _is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be a power of 2, got {rot_size}")

    # CPU fallback: use dense matmul
    if not x.is_cuda or not _HAS_TRITON:
        return _rotate_activations_cpu(x, rot_size)

    # CUDA path: select Regular vs Sylvester kernel
    orig_shape = x.shape
    orig_dtype = x.dtype

    # Flatten to 2D for the kernel
    x_flat = x.reshape(-1, x.shape[-1]).float()

    if _is_power_of_four(rot_size):
        out_flat = _fht_rotate_cuda_regular(x_flat, rot_size)
    else:
        out_flat = _fht_rotate_cuda_sylvester(x_flat, rot_size)

    # Reshape to original leading dims, with padded feature dim.
    # The kernel pads K to a multiple of rot_size internally, so
    # out_flat.shape[-1] may differ from orig_shape[-1].
    out_shape = (*orig_shape[:-1], out_flat.shape[-1])
    return out_flat.reshape(out_shape).to(orig_dtype)
