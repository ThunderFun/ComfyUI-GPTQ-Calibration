"""Per-layer activation statistics collection for GPTQ calibration.

Two phases: (1) register forward_pre_hooks that accumulate ``H += xᵀx``
and optionally ``amax``, (2) post-process: normalise, validate, crop padding.

Glossary: Hessian = unnormalised Gram matrix; amax = running max |x|;
mu2 = per-channel second moment; rot_size = Hadamard group size;
block_size = on-diagonal Hessian block size (0 = full).
"""

# ── Standard library ───────────────────────────────────────────────────────
import datetime
import gc
import logging
import math
import os
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

# ── First-party (ComfyUI) ─────────────────────────────────────────────────
import comfy.model_management
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers

# Optional: Triton FHT kernel for O(N log N) rotation on GPU.
# Falls back to the CPU matmul path when Triton/CUDA unavailable.
try:
    from kernels.triton_fht_rotate import fht_rotate as _fht_rotate
except ImportError:
    _fht_rotate = None


logger = logging.getLogger("comfyui_gptq_calibration")

__all__ = [
    "collect_stats",
    "collect_stats_dual",
    "rotate_activations",
    "get_hadamard",
    "permute_hessian",
    "ActivationStatsCollector",
    "FrequentDirections",
]


# ── Module-level constants ─────────────────────────────────────────────────

# Layer type strings used throughout the module and serialized into the
# output ``layer_types`` dict.  Centralised here to prevent typos.
_LAYER_TYPE_LINEAR: str = "Linear"
_LAYER_TYPE_CONV2D: str = "Conv2d"

# Block-diagonal Hessian threshold: only use block-gram when n_features
# exceeds block_size by this factor.  Avoids accuracy loss on small layers.
_BLOCK_GRAM_MIN_FEATURES_RATIO: int = 4

# Headroom added to FrequentDirections sketch size so a single insert after
# truncation doesn't immediately re-trigger it.
_DLR_SKETCH_SLOP: int = 4

# Corruption-detection thresholds for _validate_hessians.
# _CORRUPTION_MAX_VAL: minimum max absolute value to trigger outlier check.
# _CORRUPTION_DIAG_RATIO: max/diagonal ratio indicating outlier corruption.
# _CORRUPTION_SYM_RATIO: maximum tolerable relative asymmetry.
# _CORRUPTION_MIN_MAG: minimum magnitude to trigger asymmetry check.
_CORRUPTION_MAX_VAL: float = 1e6
_CORRUPTION_DIAG_RATIO: float = 1e3
_CORRUPTION_SYM_RATIO: float = 0.01
_CORRUPTION_MIN_MAG: float = 1e3

# CUDA cache clear interval (in samples) to prevent OOM during long runs.
_CACHE_CLEAR_INTERVAL: int = 4

# Default damping ratio advertised in calibration metadata.
_RECOMMENDED_DAMPING_RATIO: float = 0.01

# Default sampler configuration for the calibration sampling loop.
_DEFAULT_SAMPLER: str = "euler"
_DEFAULT_SCHEDULER: str = "simple"
_DEFAULT_CFG: float = 1.0

# Bitmask selecting every other bit (even positions, 0-indexed).  A power
# of 4 has exactly one 1-bit at an even position, so
# (n & _POWER_OF_FOUR_MASK) == n.
_POWER_OF_FOUR_MASK: int = 0x55555555

# Activation ceiling: skip activations whose max absolute value exceeds
# this to avoid float32 overflow when computing xᵀx.
_ACT_AMAX_CEILING: float = 1e6


# ── Utilities ──────────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration (e.g. '45s', '12m 34s', '2h 5m 3s')."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _classify_layer(module: nn.Module) -> str:
    """Return the canonical layer-type string for *module* (``'Linear'`` or ``'Conv2d'``)."""
    if isinstance(module, nn.Conv2d):
        return _LAYER_TYPE_CONV2D
    return _LAYER_TYPE_LINEAR


def _compute_eta(done: int, total: int, sample_start: float,
                 overall_start: float) -> Tuple[float, float, float]:
    """Return ``(elapsed, avg_per_sample, remaining_eta)`` in seconds."""
    elapsed = time.time() - sample_start
    total_elapsed = time.time() - overall_start
    avg = total_elapsed / done
    remaining = avg * (total - done)
    return elapsed, avg, remaining


def _resolve_mmap_dir(base_path: Optional[str]) -> Optional[str]:
    """Return a temp directory for memory-mapped Hessian files, or ``None``.

    If *base_path* is given the mmap directory is created next to it;
    otherwise ``tempfile.gettempdir()`` is used.
    """
    if base_path:
        d = os.path.join(os.path.dirname(os.path.abspath(base_path)), ".gptq_hessian_tmp")
    else:
        d = os.path.join(tempfile.gettempdir(), "gptq_hessian_tmp")
    os.makedirs(d, exist_ok=True)
    return d


def _dlr_sketch_size(rank: int) -> int:
    """Return the FrequentDirections sketch size for a given rank.

    The ``_DLR_SKETCH_SLOP`` headroom prevents immediate re-truncation
    after a single insert.
    """
    return 2 * rank + _DLR_SKETCH_SLOP


# ── ConvRot Hadamard rotation ──────────────────────────────────────────────

# Hadamard construction is O(n²) via Kronecker product — caching avoids
# redundant reconstruction when the same rot_size is used across layers.
_HADAMARD_CACHE: Dict[tuple, torch.Tensor] = {}


def _is_power_of_four(n: int) -> bool:
    """Return True if *n* is a power of 4 (4, 16, 64, 256, ...)."""
    if n < 4:
        return False
    return (n & (n - 1)) == 0 and (n & _POWER_OF_FOUR_MASK) == n


def get_hadamard(size: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Return an orthonormal Hadamard matrix of the given size (must be a power of 2).

    Power-of-4 sizes use a 4-element butterfly (fewer FHT stages);
    other powers of 2 fall back to Sylvester construction.  Cached.
    """
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


def rotate_activations(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply group-wise Hadamard rotation, padding the last dim to a multiple of ``rot_size``.

    Splits the feature dim into groups of ``rot_size`` and multiplies each
    group by the same Hadamard matrix.  Pads (not truncates) to fill the
    last group.  The caller (``collect_stats``) crops back after accumulation.
    """
    if not ((rot_size & (rot_size - 1)) == 0 and rot_size > 0):
        raise ValueError(f"rot_size must be a power of 2, got {rot_size}")
    orig_features = x.shape[-1]
    if orig_features % rot_size != 0:
        pad = rot_size - (orig_features % rot_size)
        x = torch.nn.functional.pad(x, (0, pad))
    in_features = x.shape[-1]
    num_groups = in_features // rot_size
    leading_shape = x.shape[:-1]
    x_flat = x.reshape(-1, num_groups, rot_size)
    H = get_hadamard(rot_size, x.dtype, x.device)
    x_rotated = torch.matmul(x_flat, H.T)
    return x_rotated.reshape(*leading_shape, in_features)


def permute_hessian(H: Union[Dict, List[torch.Tensor], torch.Tensor],
                    perm: torch.Tensor) -> Union[Dict, List[torch.Tensor], torch.Tensor]:
    """Permute a Hessian matrix according to a channel reordering.

    Supports: ``dict`` (DLR), ``list[Tensor]`` (block-diagonal),
    ``Tensor(dim=3)`` (stacked blocks), ``Tensor(dim=2)`` (full).
    Returns the same container type as *H*.
    """
    perm = perm.to(torch.int64)

    if isinstance(H, dict) and H.get("format") == "dlr":
        perm = perm.to(H["D"].device)
        new_H = dict(H)
        new_H["D"] = H["D"][perm]
        new_H["U"] = H["U"][perm]
        return new_H

    if isinstance(H, list):
        # ── Block-diagonal permutation ────────────────────────────────
        # Phase 1: build global_index → (block, local_index) lookup tables.
        # Phase 2: for each output block, gather entries whose row and col
        # originate in the same source block (cross-block entries are zero).
        # List processing builds CPU helper tensors — move perm to CPU.
        perm = perm.cpu()
        block_sizes = [b.shape[0] for b in H]
        num_blocks = len(block_sizes)
        offsets = [0]
        for bs in block_sizes:
            offsets.append(offsets[-1] + bs)
        n = offsets[-1]
        perm_n = perm.size(0)

        # perm may be smaller than n when PermuQuant crops padding
        # (e.g. convrot with non-divisible in_features).  Only permute
        # blocks that fall within [0, perm_n).
        block_of = torch.empty(perm_n, dtype=torch.int64)
        local_of = torch.empty(perm_n, dtype=torch.int64)
        filled = 0
        for bi, bs in enumerate(block_sizes):
            take = min(bs, perm_n - filled)
            if take <= 0:
                break
            block_of[filled:filled + take] = bi
            local_of[filled:filled + take] = torch.arange(take)
            filled += take

        perm_block = block_of[perm]
        perm_local = local_of[perm]

        # Build a mask for each block to select only rows that fall within
        # perm_n (avoids out-of-bounds slicing when perm_n < n).
        block_mask = torch.arange(perm_n)
        h_device = H[0].device

        new_blocks = []
        for i in range(num_blocks):
            ri = offsets[i]
            bs_i = block_sizes[i]
            # Rows in the perm that map to this output block's range.
            mask = (block_mask >= ri) & (block_mask < ri + bs_i)
            if not mask.any():
                continue  # padding block — drop entirely

            rb = perm_block[mask]
            rl = perm_local[mask]
            actual_size = rb.size(0)

            block = torch.zeros(actual_size, actual_size, dtype=H[0].dtype, device=h_device)
            for li in range(actual_size):
                k = rb[li].item()
                same = (rb == k)
                if same.any():
                    block[li, same] = H[k][rl[li], rl[same]]
            new_blocks.append(block)

        return new_blocks

    if isinstance(H, torch.Tensor):
        perm = perm.to(H.device)
        if H.dim() == 2:
            return H[perm][:, perm]
        if H.dim() == 3:
            block_size = H.shape[1]
            num_blocks = H.shape[0]
            n = num_blocks * block_size
            H_full = torch.zeros(n, n, dtype=H.dtype, device=H.device)
            for i in range(num_blocks):
                start = i * block_size
                H_full[start:start + block_size, start:start + block_size] = H[i]
            H_perm = H_full[perm][:, perm]
            del H_full
            blocks = []
            for i in range(num_blocks):
                start = i * block_size
                blocks.append(H_perm[start:start + block_size, start:start + block_size])
            return torch.stack(blocks)

    return H


class FrequentDirections:
    """Streaming sketch for Diagonal + Low-Rank (DLR) Hessian approximation.

    Inspired by Frequent Directions (Liberty, KDD 2013), but uses plain
    truncated SVD (keeps top-ℓ/2 singular values, no σ² subtraction).
    Here ℓ = :attr:`sketch_size`.  An exact per-channel diagonal ``_diag``
    is accumulated separately and used to form the residual diagonal in
    :meth:`dlr_decompose`.
    """

    def __init__(self, n: int, sketch_size: int, rank: int, device=None):
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if sketch_size < 2 * rank:
            raise ValueError(
                f"sketch_size ({sketch_size}) must be >= 2 * rank ({2 * rank})"
            )
        self.n = n
        self.sketch_size = sketch_size
        self.rank = rank
        _dev = device or "cpu"
        self.sketch = torch.zeros(sketch_size, n, dtype=torch.float32, device=_dev)
        self._next_row = 0
        self._diag = torch.zeros(n, dtype=torch.float32, device=_dev)

    def update(self, x: torch.Tensor) -> None:
        """Absorb rows of *x* into the sketch (1-D or 2-D)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.float()

        remaining = x
        while remaining.shape[0] > 0:
            available = self.sketch_size - self._next_row
            if available == 0:
                self._truncate()
                available = self.sketch_size - self._next_row
            take = min(available, remaining.shape[0])
            batch = remaining[:take]
            self.sketch[self._next_row:self._next_row + take] = batch
            self._next_row += take
            self._diag += (batch ** 2).sum(dim=0)
            remaining = remaining[take:]

    def _truncate(self) -> None:
        """SVD truncation: keep top-ℓ/2 singular values, zero the rest."""
        active = self.sketch[:self._next_row]
        if active.shape[0] == 0:
            return

        U, S, Vh = torch.linalg.svd(active, full_matrices=False)
        keep = min(self.sketch_size // 2, S.shape[0])

        if keep == 0:
            self.sketch.zero_()
            self._next_row = 0
            return

        new_rows = S[:keep].unsqueeze(1) * Vh[:keep, :]  # (keep, n)
        self.sketch[:keep] = new_rows
        self.sketch[keep:] = 0
        self._next_row = keep

    def dlr_decompose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(D, U)`` such that ``D + UUᵀ ≈ XᵀX``.

        ``D`` is the residual diagonal (``_diag − diag(UUᵀ)``, clamped ≥ 0)
        and ``U`` is ``(n, r)`` top-r eigenvectors scaled by ``√λ``.
        """
        if self._next_row == 0:
            r = self.rank
            return self._diag.clone(), torch.zeros(self.n, r, dtype=torch.float32, device=self.sketch.device)

        active = self.sketch[:self._next_row]
        U_svd, S_svd, Vh = torch.linalg.svd(active, full_matrices=False)
        r = min(self.rank, S_svd.shape[0])
        eigvals = (S_svd[:r] ** 2).clamp(min=0)
        U = Vh[:r].T * eigvals.sqrt().unsqueeze(0)  # (n, r)

        # Pad U to (n, rank) with zero columns for stable shape.
        if r < self.rank:
            pad = torch.zeros(self.n, self.rank - r, dtype=torch.float32, device=self.sketch.device)
            U = torch.cat([U, pad], dim=1)

        # D = exact diagonal − diag(UUᵀ), clamped ≥ 0 for sketch error.
        uu_diag = (U ** 2).sum(dim=1)  # diag(UUᵀ) = Σ_k U_ik²
        D = (self._diag - uu_diag).clamp(min=0)

        return D, U


class ActivationStatsCollector:
    """Forward pre-hook that accumulates Hessian and/or amax stats for one layer.

    Multiple collectors share a single ``store`` dict and accumulate
    results in-place.  Modes: ``"hessian"`` (default), ``"mu2"`` (second
    moments for PermuQuant), ``"both"`` (both in one pass).
    """

    def __init__(self,
                 layer_name: str,
                 store: Dict,
                 layer_type: str,
                 hessian_block_size: int = 0,
                 collect_amax: bool = True,
                 rot_size: int = 0,
                 mode: str = "hessian",
                 permutation: Optional[torch.Tensor] = None,
                 hessian_format: str = "block",
                 dlr_rank: int = 0,
                 force_cpu: bool = False):
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.store = store
        self.hessian_block_size = int(hessian_block_size)
        self.collect_amax = bool(collect_amax)
        self.rot_size = int(rot_size)
        self.mode = mode
        self.permutation = permutation
        self.hessian_format = hessian_format
        self.dlr_rank = int(dlr_rank)
        self.force_cpu = bool(force_cpu)
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        self._nan_skip_count: int = 0

    def register(self, module: nn.Module) -> None:
        """Attach a ``forward_pre_hook`` to *module* that accumulates stats."""
        handle = module.register_forward_pre_hook(self._hook_fn)
        self.hooks.append(handle)

    def remove(self) -> None:
        """Remove all hooks and finalize any DLR sketch."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self._finalize_dlr()

    def _finalize_dlr(self) -> None:
        """Decompose FD sketch into a DLR dict. Called by :meth:`remove`."""
        if self.hessian_format != "dlr":
            return
        hessians = self.store.get("hessians", {})
        H = hessians.get(self.layer_name)
        if isinstance(H, FrequentDirections):
            D, U = H.dlr_decompose()
            hessians[self.layer_name] = {
                "format": "dlr",
                "D": D,
                "U": U,
                "rank": int(U.shape[1]),
                "n": int(D.shape[0]),
            }

    # ---- internal helpers ------------------------------------------------

    def _use_block_gram(self, n_features: int) -> bool:
        """Block-diagonal only when n_features is much larger than block_size.

        The ``_BLOCK_GRAM_MIN_FEATURES_RATIO`` threshold avoids accuracy
        loss on small layers (same ratio as in ``_allocate_layer_buffers``).
        """
        return (self.hessian_block_size > 0
                and n_features > self.hessian_block_size * _BLOCK_GRAM_MIN_FEATURES_RATIO)

    def _make_fd(self, n_features: int, device=None) -> FrequentDirections:
        """Create a FrequentDirections sketch sized for this layer."""
        rank = min(self.dlr_rank, n_features)
        return FrequentDirections(
            n=n_features, sketch_size=_dlr_sketch_size(rank), rank=rank,
            device=device,
        )

    def _flatten_linear_input(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten to ``(rows, in_features)``."""
        return x.reshape(-1, x.shape[-1])

    def _flatten_conv_input(self, x: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
        """Unfold Conv2d to ``(patches, in_channels·kH·kW)`` — matches GPTQ's flattened view."""
        unfold = nn.Unfold(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
        )
        patches = unfold(x)
        patches = patches.permute(0, 2, 1)
        return patches.reshape(-1, patches.shape[-1])

    def _accumulate_hessian(self, x_flat: torch.Tensor) -> None:
        """Accumulate ``xᵀx`` into the Hessian for this layer.

        Dispatches to the correct accumulator based on the current Hessian
        storage format (DLR sketch, block-diagonal list, stacked tensor,
        or full dense/mmap tensor).
        """
        hessians = self.store.setdefault("hessians", {})
        row_counts = self.store.setdefault("_hessian_row_count", {})
        n_rows = x_flat.shape[0]

        existing = hessians.get(self.layer_name)
        if existing is None:
            self._init_hessian(hessians, x_flat)
        else:
            self._update_hessian(existing, x_flat)

        row_counts[self.layer_name] = row_counts.get(self.layer_name, 0) + n_rows

    def _init_hessian(self, hessians: Dict, x_flat: torch.Tensor) -> None:
        """Allocate and initialize the Hessian store for this layer (first call)."""
        if self.hessian_format == "dlr":
            fd = self._make_fd(x_flat.shape[1], device=x_flat.device)
            fd.update(x_flat)
            hessians[self.layer_name] = fd
        elif self._use_block_gram(x_flat.shape[1]):
            hessians[self.layer_name] = self._block_gram(x_flat, self.hessian_block_size)
        else:
            hessians[self.layer_name] = x_flat.T @ x_flat

    def _update_hessian(self, existing, x_flat: torch.Tensor) -> None:
        """Update an existing Hessian store with new activations."""
        if isinstance(existing, FrequentDirections):
            existing.update(x_flat)
        elif isinstance(existing, list):
            self._update_block_list(existing, x_flat)
        elif existing.dim() == 3:
            self._update_stacked_tensor(existing, x_flat)
        else:
            # Full Hessian (Tensor, possibly mmap'd on CPU).
            # When force_cpu=False the activation is on GPU but the mmap
            # buffer is on CPU — move the activation to match.
            if x_flat.device != existing.device:
                x_flat = x_flat.to(existing.device)
            existing.add_(x_flat.T @ x_flat)

    def _update_block_list(self, blocks: List[torch.Tensor], x_flat: torch.Tensor) -> None:
        """Accumulate xᵀx into a block-diagonal list of tensors."""
        block_size = self.hessian_block_size
        for i, block in enumerate(blocks):
            start = i * block_size
            end = min((i + 1) * block_size, x_flat.shape[1])
            if start >= x_flat.shape[1]:
                break
            xi = x_flat[:, start:end]
            block.add_(xi.T @ xi)

    def _update_stacked_tensor(self, existing: torch.Tensor, x_flat: torch.Tensor) -> None:
        """Accumulate xᵀx into a stacked 3-D tensor (backward-compat format)."""
        block_size = existing.shape[1]
        num_blocks = existing.shape[0]
        n = x_flat.shape[1]
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, n)
            xi = x_flat[:, start:end]
            width = end - start
            if width < block_size:
                xi = torch.nn.functional.pad(xi, (0, block_size - width))
            existing[i].add_(xi.T @ xi)

    def _block_gram(self, x: torch.Tensor, block_size: int) -> List[torch.Tensor]:
        """Diagonal blocks of xᵀx.  Last block retains its true size (not zero-padded)."""
        n = x.shape[1]
        num_blocks = (n + block_size - 1) // block_size
        blocks = []
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, n)
            xi = x[:, start:end]
            blocks.append(xi.T @ xi)
        return blocks

    def _accumulate_amax(self, x_flat: torch.Tensor) -> None:
        """Track running max |x| for this layer."""
        if not self.collect_amax:
            return
        amax_store = self.store.setdefault("amax", {})
        current = x_flat.abs().max().item()
        # NaN > prev is False in Python, so a NaN ``current`` would silently
        # fail to update the store — masking the fact that the layer saw
        # non-finite activations.  Use math.isnan to detect this and skip.
        if math.isnan(current):
            return
        prev = amax_store.get(self.layer_name)
        if prev is None or current > prev:
            amax_store[self.layer_name] = current

    def _accumulate_mu2(self, x_flat: torch.Tensor) -> None:
        """Accumulate per-channel second moments for PermuQuant."""
        mu2_store = self.store.setdefault("mu2", {})
        count_store = self.store.setdefault("_mu2_count", {})
        mu2 = (x_flat ** 2).sum(dim=0)  # [in_features]
        prev_sum = mu2_store.get(self.layer_name)
        if prev_sum is None:
            mu2_store[self.layer_name] = mu2
        else:
            prev_sum.add_(mu2)
        count_store[self.layer_name] = count_store.get(self.layer_name, 0) + x_flat.shape[0]

    # ---- hook entry point ------------------------------------------------

    def _hook_fn(self, module: nn.Module, inputs: tuple) -> None:
        """``forward_pre_hook``: detach, flatten, guard, rotate, accumulate.

        This hook is side-effect-free with respect to *inputs* — it only
        reads ``inputs[0]`` and never modifies the tensor.  All mutations
        are to ``self.store``.

        Pipeline: detach → flatten → float32 → NaN guard → rotate → accumulate.
        GPU-fast by default; ``force_cpu=True`` moves everything to CPU.
        """
        if not inputs:
            return
        x = inputs[0].detach()
        if self.force_cpu:
            x = x.cpu()
        if isinstance(module, nn.Conv2d):
            x_flat = self._flatten_conv_input(x, module)
        else:
            x_flat = self._flatten_linear_input(x)

        if x_flat.numel() == 0:
            return

        # Cast to float32 on input device (GPU when force_cpu=False)
        x_flat = x_flat.float()

        # ── NaN/Inf guard ─────────────────────────────────────────────
        # Skip non-finite activations before (and after) rotation.
        if not torch.isfinite(x_flat).all():
            self._nan_skip_count += 1
            return

        # Outlier guard: skip activations that would overflow float32 when squared.
        if x_flat.abs().max().item() > _ACT_AMAX_CEILING:
            self._nan_skip_count += 1
            return

        if self.rot_size > 0:
            if x_flat.is_cuda and _fht_rotate is not None:
                # GPU path: O(N log N) FHT on CUDA.
                x_flat = _fht_rotate(x_flat, self.rot_size)
            else:
                # CPU path: O(N²) matmul with Hadamard matrix.
                x_flat = rotate_activations(x_flat, self.rot_size)

        # Defensive post-rotation NaN check.
        if not torch.isfinite(x_flat).all():
            self._nan_skip_count += 1
            return

        if self.mode == "mu2":
            self._accumulate_mu2(x_flat)
            return

        if self.mode == "both":
            self._accumulate_mu2(x_flat)
            self._accumulate_hessian(x_flat)
            self._accumulate_amax(x_flat)
            return

        if self.permutation is not None:
            x_flat = x_flat[:, self.permutation]

        self._accumulate_hessian(x_flat)
        self._accumulate_amax(x_flat)


def _walk_target_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for every Linear/Conv2d in *model*."""
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            targets.append((name, module))
    return targets


def _get_in_features(module: nn.Module, ltype: str) -> int:
    """Return the flattened input feature count for a target layer."""
    if ltype == _LAYER_TYPE_CONV2D:
        return module.in_channels * module.kernel_size[0] * module.kernel_size[1]
    return module.in_features


def _allocate_hessian(
    layer_name: str,
    shape: tuple,
    use_mmap: bool,
    mmap_dir: Optional[str],
    store: dict,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a zero-initialized Hessian tensor, optionally memory-mapped to disk.

    Full Hessians for large layers can exceed GPU RAM (e.g. 16k×16k = 1 GiB).
    mmap keeps them on disk, freeing GPU memory for the model.
    """
    if not use_mmap:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    if mmap_dir is None:
        raise RuntimeError("mmap_dir must be provided when use_mmap=True")
    os.makedirs(mmap_dir, exist_ok=True)
    safe_name = layer_name.replace("/", "_").replace(".", "_").replace(" ", "_")
    path = os.path.join(mmap_dir, f"{safe_name}.hessian")
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    mm[:] = 0.0
    mm.flush()
    t = torch.from_numpy(mm)
    # Keep the numpy memmap alive so PyTorch does not free the underlying buffer
    store.setdefault("_mmap_refs", []).append(mm)
    return t


def _resolve_inner_model(model_patcher) -> nn.Module:
    """Return the underlying ``nn.Module`` to walk for layer hooks."""
    inner = getattr(model_patcher, "model", None)
    if inner is None:
        raise ValueError("model_patcher has no .model attribute")
    diffusion = getattr(inner, "diffusion_model", None)
    if diffusion is not None:
        return diffusion
    return inner


def _infer_latent_shape(model_patcher, height: int = 64, width: int = 64) -> Tuple[int, int, int]:
    """Return ``(channels, H, W)`` for the calibration noise tensor."""
    inner = getattr(model_patcher, "model", None)
    if inner is None:
        return (4, int(height), int(width))
    latent_format = getattr(inner, "latent_format", None)
    channels = getattr(latent_format, "latent_channels", 4) if latent_format is not None else 4
    return (int(channels), int(height), int(width))


def _resolve_model_sampling(model_patcher) -> Optional[object]:
    """Return the model_sampling object, or None."""
    inner = getattr(model_patcher, "model", None)
    return getattr(inner, "model_sampling", None) if inner is not None else None


def _validate_sigma_clip(sigma_min: float, sigma_max: float) -> None:
    """Validate sigma clipping parameters."""
    if sigma_min > sigma_max:
        raise ValueError(
            f"sigma_min ({sigma_min}) must be <= sigma_max ({sigma_max})"
        )
    if sigma_min == sigma_max:
        raise ValueError(
            f"sigma_min ({sigma_min}) == sigma_max ({sigma_max}); "
            f"at least one sampling step is required (sigma_min < sigma_max)."
        )
    if sigma_min < 0.0:
        raise ValueError(f"sigma_min ({sigma_min}) must be >= 0.0")
    if sigma_max > 1.0:
        raise ValueError(f"sigma_max ({sigma_max}) must be <= 1.0")


def _build_sigmas(model_sampling, num_steps: int,
                  sigma_min: float = 0.0, sigma_max: float = 1.0) -> torch.Tensor:
    """Return ``num_steps + 1`` sigmas ending at 0, clamped to [sigma_min, sigma_max]."""
    if sigma_min != 0.0 or sigma_max != 1.0:
        _validate_sigma_clip(sigma_min, sigma_max)

    if model_sampling is not None:
        try:
            sigmas = comfy.samplers.simple_scheduler(model_sampling, num_steps)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("simple_scheduler failed (%s); falling back to linear", exc)
            sigmas = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    else:
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)

    # ── Sigma-range clipping ─────────────────────────────────────────
    if sigma_min != 0.0 or sigma_max != 1.0:
        sigmas = sigmas.clamp(min=sigma_min, max=sigma_max)
        # Append sigma_min so the schedule reaches its lower bound.
        # When sigma_min == 0 the default schedule already ends at 0.
        if sigmas[-1].item() > sigma_min:
            sigmas = torch.cat([sigmas, torch.tensor([sigma_min], dtype=sigmas.dtype)])
        # Deduplicate adjacent equal sigmas (clamp produces runs of identical values).
        sigmas = torch.unique_consecutive(sigmas)
        if sigmas.shape[0] < 2:
            sigmas = torch.tensor([sigma_max, sigma_min], dtype=torch.float32)
        effective = sigmas.shape[0] - 1
        if effective != num_steps:
            logger.info(
                "Sigma clipping [%s, %s]: effective steps = %d (requested %d)",
                sigma_min, sigma_max, effective, num_steps,
            )

    return sigmas.to(torch.float32)


def _validate_hessians(hessians: Dict[str, torch.Tensor]) -> None:
    """Check Hessians for corruption (extreme outliers, asymmetry).

    Logs warnings only — does not raise or modify the Hessians.
    Three heuristics are applied:
    1. Non-finite values or negative diagonal (DLR format).
    2. Off-diagonal outlier: ``max_val / diag_max > _CORRUPTION_DIAG_RATIO``
       when ``max_val > _CORRUPTION_MAX_VAL`` (full Hessian).
    3. Asymmetry: ``sym_err > max_val * _CORRUPTION_SYM_RATIO``
       when ``max_val > _CORRUPTION_MIN_MAG``.
    """
    corrupted = []
    for name, H in hessians.items():
        if isinstance(H, dict) and H.get("format") == "dlr":
            D = H["D"]
            U = H["U"]
            if not torch.isfinite(D).all() or not torch.isfinite(U).all():
                corrupted.append(name)
                logger.warning("Non-finite values in DLR Hessian %s", name)
            elif D.min().item() < 0:
                corrupted.append(name)
                logger.warning(
                    "Negative diagonal in DLR Hessian %s: min=%.6f", name, D.min().item(),
                )
            continue

        if not isinstance(H, torch.Tensor):
            continue

        if H.dim() == 2:
            # Full Hessian — check symmetry and outlier elements
            sym_err = (H - H.T).abs().max().item()
            max_val = H.abs().max().item()
            diag_max = H.diagonal().abs().max().item()

            # Off-diagonal > _CORRUPTION_DIAG_RATIO × diagonal → outlier corruption
            if (max_val > _CORRUPTION_MAX_VAL and diag_max > 0
                    and max_val / diag_max > _CORRUPTION_DIAG_RATIO):
                corrupted.append(name)
                logger.warning(
                    "Hessian corruption detected in %s: max=%.0f, diag_max=%.0f, ratio=%.0f",
                    name, max_val, diag_max, max_val / diag_max,
                )
            elif sym_err > max_val * _CORRUPTION_SYM_RATIO and max_val > _CORRUPTION_MIN_MAG:
                corrupted.append(name)
                logger.warning(
                    "Hessian asymmetry detected in %s: sym_err=%.0f, max=%.0f",
                    name, sym_err, max_val,
                )

        elif H.dim() == 3 or isinstance(H, list):
            # Block-diagonal — check each block for symmetry
            block_list = H if isinstance(H, list) else [H[bi] for bi in range(H.shape[0])]
            for bi, block in enumerate(block_list):
                sym_err = (block - block.T).abs().max().item()
                max_val = block.abs().max().item()
                if sym_err > max_val * _CORRUPTION_SYM_RATIO and max_val > _CORRUPTION_MIN_MAG:
                    corrupted.append(f"{name}[block {bi}]")
                    logger.warning(
                        "Hessian block asymmetry in %s block %d: sym_err=%.0f",
                        name, bi, sym_err,
                    )

    if corrupted:
        logger.warning(
            "%d Hessians flagged as potentially corrupted. "
            "Consider re-running calibration with a different seed.", len(corrupted),
        )


def _run_samples(model_patcher, conditioning, num_steps: int, num_samples: int,
                 seed: int, latent_height: int, latent_width: int,
                 progress_callback: Optional[Callable[[int, int, str], None]],
                 pass_label: str = "", sigma_min: float = 0.0,
                 sigma_max: float = 1.0) -> None:
    """Run the calibration sampling loop for a single model."""
    setup_start = time.time()
    device = model_patcher.load_device
    dtype = torch.float32
    channels, lat_h, lat_w = _infer_latent_shape(model_patcher, latent_height, latent_width)
    batch = 1

    sigmas = _build_sigmas(_resolve_model_sampling(model_patcher), num_steps,
                           sigma_min=sigma_min, sigma_max=sigma_max).to(device)
    positive = conditioning
    negative = conditioning
    logger.info("model setup (%s): %.2fs", pass_label, time.time() - setup_start)

    total = max(1, int(num_samples))
    overall_start = time.time()
    for sample_idx in range(total):
        sample_start = time.time()
        # Re-seed per sample so each sample is independently reproducible
        # (matches the dual-model seeding strategy).
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + sample_idx)
        noise = torch.randn(
            (batch, channels, lat_h, lat_w),
            generator=generator,
            dtype=dtype,
        ).to(device)

        try:
            comfy.sample.sample(
                model=model_patcher,
                noise=noise,
                steps=num_steps,
                cfg=_DEFAULT_CFG,
                sampler_name=_DEFAULT_SAMPLER,
                scheduler=_DEFAULT_SCHEDULER,
                positive=positive,
                negative=negative,
                latent_image=noise,
                denoise=1.0,
                sigmas=sigmas,
                disable_pbar=True,
                seed=int(seed) + sample_idx,
            )
        except Exception as exc:
            logger.exception("Sample %d/%d failed: %s", sample_idx + 1, total, exc)
            raise

        done = sample_idx + 1
        elapsed, avg, remaining = _compute_eta(done, total, sample_start, overall_start)
        msg = f"sample {done}/{total} ({_fmt_duration(elapsed)}, avg {_fmt_duration(avg)}, ETA {_fmt_duration(remaining)})"
        if progress_callback is not None:
            try:
                progress_callback(done, total, msg)
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)

        # Periodically clear CUDA cache to prevent OOM during long runs
        if (done % _CACHE_CLEAR_INTERVAL) == 0 and torch.cuda.is_available():
            comfy.model_management.soft_empty_cache()

    total_elapsed = time.time() - overall_start
    logger.info("Calibration (%s) complete: %d samples in %.1fs", pass_label, total, total_elapsed)


def _extract_hessian_diag(H) -> Optional[torch.Tensor]:
    """Extract the diagonal from a Hessian (full, block-diagonal, or DLR).

    Returns a clone (float32) for DLR and contiguous tensors, or a new
    concatenated tensor for block-diagonal formats.  Returns None when
    the format is unrecognised.
    """
    if isinstance(H, dict) and H.get("format") == "dlr":
        return H["D"].clone().float()
    if isinstance(H, list):
        parts = []
        for block in H:
            if isinstance(block, torch.Tensor) and block.dim() == 2:
                parts.append(block.diagonal())
        if not parts:
            return None
        return torch.cat(parts).float()
    if isinstance(H, torch.Tensor):
        if H.dim() == 2:
            return H.diagonal().float()
        if H.dim() == 3:
            return torch.cat([H[i].diagonal() for i in range(H.shape[0])]).float()
    return None


# ── Helpers shared by collect_stats and collect_stats_dual ──────────────────


def _allocate_layer_buffers(
    model_patcher,
    store: Dict,
    hessian_block_size: int,
    collect_amax: bool,
    rot_size: int,
    permuquant: bool,
    mmap_dir: Optional[str],
    use_mmap_full: bool,
    alloc_start: float = 0.0,
    hessian_format: str = "block",
    dlr_rank: int = 0,
    device: str = "cpu",
    force_cpu: bool = False,
) -> Tuple[List[ActivationStatsCollector], List[Tuple[str, nn.Module]], Dict, Dict]:
    """Walk target modules, allocate buffers, and register hooks.

    Returns ``(collectors, targets, shapes, layer_types)``.
    """
    target = _resolve_inner_model(model_patcher)
    targets = _walk_target_modules(target)

    # Hessian buffers live on GPU when hooks run on GPU (force_cpu=False),
    # except mmap'd full Hessians which are always on CPU.
    hessian_device = "cpu" if force_cpu else device

    collector_mode = "both" if permuquant else "hessian"
    shapes: Dict = {}
    layer_types: Dict = {}
    collectors: List[ActivationStatsCollector] = []

    for idx, (name, module) in enumerate(targets, 1):
        ltype = _classify_layer(module)
        layer_types[name] = ltype
        shapes[name] = tuple(module.weight.shape)

        in_f = _get_in_features(module, ltype)
        alloc_f = in_f
        if rot_size > 0 and in_f % rot_size != 0:
            alloc_f = rot_size * ((in_f + rot_size - 1) // rot_size)

        if hessian_format == "dlr" and dlr_rank > 0:
            rank = min(dlr_rank, alloc_f)
            store["hessians"][name] = FrequentDirections(
                n=alloc_f, sketch_size=_dlr_sketch_size(rank), rank=rank,
                device=hessian_device,
            )
        elif hessian_format != "full" and hessian_block_size > 0 and alloc_f > hessian_block_size * _BLOCK_GRAM_MIN_FEATURES_RATIO:
            num_blocks = (alloc_f + hessian_block_size - 1) // hessian_block_size
            blocks = []
            for i in range(num_blocks):
                start = i * hessian_block_size
                end = min((i + 1) * hessian_block_size, alloc_f)
                width = end - start
                blocks.append(torch.zeros(width, width, dtype=torch.float32,
                                          device=hessian_device))
            store["hessians"][name] = blocks
        else:
            H_shape = (alloc_f, alloc_f)
            # _allocate_hessian keeps mmap buffers on CPU regardless
            store["hessians"][name] = _allocate_hessian(
                name, H_shape, use_mmap_full, mmap_dir, store,
                device=hessian_device,
            )

        collector = ActivationStatsCollector(
            layer_name=name,
            store=store,
            layer_type=ltype,
            hessian_block_size=hessian_block_size,
            collect_amax=collect_amax,
            rot_size=rot_size,
            mode=collector_mode,
            hessian_format=hessian_format,
            dlr_rank=dlr_rank,
            force_cpu=force_cpu,
        )
        collector.register(module)
        collectors.append(collector)

        if use_mmap_full and idx % max(1, len(targets) // 10) == 0:
            logger.info(
                "Allocated %d/%d Hessian buffers (%.1fs)",
                idx, len(targets), time.time() - alloc_start,
            )

    logger.info(
        "Allocated %d Hessian buffers in %.1fs (device=%s, force_cpu=%s)",
        len(collectors), time.time() - alloc_start, hessian_device, force_cpu,
    )
    return collectors, targets, shapes, layer_types


# ── _finalize_store and its helpers ────────────────────────────────────────
#
# The ordering of these phases is critical:
#   1. NaN-skip report   (read-only on collectors)
#   2. Normalise         (mutates hessians)
#   3. PermuQuant        (pops mu2, mutates hessians — must follow normalise)
#   4. Validate          (read-only on hessians — must follow permuquant)
#   5. PiSO diagonal     (read-only on hessians)
#   6. Crop padding      (mutates hessians)
#   7. Build result      (assembles final dict — needs all of the above)


def _log_nan_skip_report(
    collectors: List[ActivationStatsCollector],
) -> Tuple[Dict[str, int], int, int]:
    """Log and return NaN-skip statistics.

    Returns ``(nan_skip_store, total_nan_skips, layers_with_skips)``.
    """
    nan_skip_store: Dict[str, int] = {}
    total_nan_skips = 0
    layers_with_skips = 0
    for c in collectors:
        if c._nan_skip_count > 0:
            nan_skip_store[c.layer_name] = c._nan_skip_count
            total_nan_skips += c._nan_skip_count
            layers_with_skips += 1
    if layers_with_skips > 0:
        logger.warning(
            "NaN/Inf detected in activations — skipped accumulation for "
            "%d layer-passes across %d/%d layers.  The Hessian and amax "
            "for affected layers reflect only the clean (finite) forward "
            "passes.  Layers with most skips: %s",
            total_nan_skips, layers_with_skips, len(collectors),
            ", ".join(
                f"{name} ({count})"
                for name, count in sorted(
                    nan_skip_store.items(), key=lambda x: -x[1]
                )[:5]
            ),
        )
    else:
        logger.info("No NaN/Inf detected in any activations during calibration")
    return nan_skip_store, total_nan_skips, layers_with_skips


def _normalize_hessians_by_row_count(store: Dict) -> None:
    """Divide each Hessian by its row count to produce a mean Gram matrix."""
    row_counts = store.pop("_hessian_row_count", {})
    hessians = store.get("hessians", {})
    for name, H in hessians.items():
        n = row_counts.get(name, 0)
        if n <= 0:
            continue
        if isinstance(H, dict) and H.get("format") == "dlr":
            H["D"].div_(n)
            H["U"].div_(math.sqrt(n))
        elif isinstance(H, list):
            for block in H:
                block.div_(n)
        elif isinstance(H, torch.Tensor):
            H.div_(n)
    if row_counts:
        logger.info(
            "Normalised Hessians by row count (median rows=%d)",
            sorted(row_counts.values())[len(row_counts) // 2],
        )


def _compute_permuquant_permutations(
    store: Dict, shapes: Dict, layer_types: Dict,
) -> Dict[str, torch.Tensor]:
    """Compute channel permutations from second-moment statistics.

    Pops ``mu2`` and ``_mu2_count`` from *store*.
    """
    mu2_store = store.pop("mu2", {})
    count_store = store.pop("_mu2_count", {})
    permutations: Dict[str, torch.Tensor] = {}
    for name, mu2_sum in mu2_store.items():
        n = count_store.get(name, 1)
        mu2 = mu2_sum / n
        shape = shapes[name]
        ltype = layer_types[name]
        if ltype == _LAYER_TYPE_CONV2D:
            in_f = shape[1] * shape[2] * shape[3]
        else:
            in_f = shape[1]
        mu2 = mu2[:in_f]
        perm = mu2.argsort(descending=True).to(torch.int32)
        permutations[name] = perm
    logger.info("PermuQuant: computed permutations for %d layers", len(permutations))
    return permutations


def _apply_permuquant_to_hessians(store: Dict, permutations: Dict[str, torch.Tensor]) -> None:
    """Apply channel permutations to Hessians in-place."""
    hessians = store.get("hessians", {})
    for name, perm in permutations.items():
        if name in hessians:
            old = hessians[name]
            hessians[name] = permute_hessian(old, perm)
            del old
    gc.collect()
    logger.info("PermuQuant: applied permutations to Hessians")


def _extract_piso_diagonals(store: Dict) -> Dict[str, torch.Tensor]:
    """Extract Hessian diagonals for PiSO data-aware scale optimization."""
    hessian_diag_store: Dict[str, torch.Tensor] = {}
    hessians = store.get("hessians", {})
    for name, H in hessians.items():
        diag = _extract_hessian_diag(H)
        if diag is not None:
            hessian_diag_store[name] = diag
    logger.info("PiSO: extracted Hessian diagonal for %d layers", len(hessian_diag_store))
    return hessian_diag_store


def _crop_rotation_padding(
    store: Dict, targets: List[Tuple[str, nn.Module]], rot_size: int,
) -> None:
    """Crop Hessians from padded size back to the layer's true in_features."""
    if rot_size <= 0:
        return
    for name, module in targets:
        ltype = _classify_layer(module)
        in_f = _get_in_features(module, ltype)
        H = store.get("hessians", {}).get(name)
        if H is None:
            continue
        if isinstance(H, dict) and H.get("format") == "dlr":
            if H["D"].shape[0] > in_f:
                H["D"] = H["D"][:in_f].clone()
                H["U"] = H["U"][:in_f].clone()
                H["n"] = in_f
        elif isinstance(H, list):
            total = sum(b.shape[0] for b in H)
            if total > in_f:
                remaining = in_f
                cropped = []
                for b in H:
                    bs = b.shape[0]
                    if remaining >= bs:
                        cropped.append(b)
                        remaining -= bs
                    elif remaining > 0:
                        cropped.append(b[:remaining, :remaining])
                        remaining = 0
                store["hessians"][name] = cropped
        elif isinstance(H, torch.Tensor):
            if H.dim() == 2 and H.shape[0] > in_f:
                store["hessians"][name] = H[:in_f, :in_f]
            elif H.dim() == 3:
                total = H.shape[0] * H.shape[1]
                if total > in_f:
                    bs = H.shape[1]
                    num_full = in_f // bs
                    remainder = in_f % bs
                    blocks = [H[i] for i in range(num_full)]
                    if remainder > 0:
                        blocks.append(H[num_full][:remainder, :remainder])
                    store["hessians"][name] = blocks


def _build_result_metadata(
    model_name: str,
    num_samples: int,
    num_steps: int,
    seed: int,
    hessian_block_size: int,
    hessian_format: str,
    dlr_rank: int,
    collect_amax: bool,
    rot_size: int,
    permuquant: bool,
    piso: bool,
    hessian_diag_store: Dict[str, torch.Tensor],
    sigma_min: float,
    sigma_max: float,
    targets: list,
    latent_shape: Tuple[int, int, int],
    layers_with_skips: int,
    total_nan_skips: int,
) -> Dict:
    """Build the metadata dict for the calibration result."""
    channels, lat_h, lat_w = latent_shape
    return {
        "model_name": model_name,
        "num_samples": int(num_samples),
        "num_steps": int(num_steps),
        "seed": int(seed),
        "hessian_block_size": int(hessian_block_size),
        "hessian_format": hessian_format,
        "dlr_rank": int(dlr_rank) if hessian_format == "dlr" else 0,
        "collect_amax": bool(collect_amax),
        "rot_size": int(rot_size),
        "hessian_rotated": rot_size > 0,
        "permuquant_enabled": permuquant,
        "piso_enabled": piso and bool(hessian_diag_store),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "collection_date": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "recommended_damping_ratio": _RECOMMENDED_DAMPING_RATIO,
        "quantization_order": "column",
        "num_layers": len(targets),
        "latent_shape": [int(channels), int(lat_h), int(lat_w)],
        "nan_skip_layers": layers_with_skips,
        "nan_skip_total": total_nan_skips,
    }


def _assemble_result(
    store: Dict,
    metadata: Dict,
    shapes: Dict,
    layer_types: Dict,
    collect_amax: bool,
    nan_skip_store: Dict[str, int],
    permutations: Dict[str, torch.Tensor],
    hessian_diag_store: Dict[str, torch.Tensor],
    mmap_dir: Optional[str],
) -> Dict:
    """Assemble the final result dict from store contents and metadata."""
    result = {
        "metadata": metadata,
        "hessians": store.get("hessians", {}),
        "amax": store.get("amax", {}) if collect_amax else {},
        "shapes": shapes,
        "layer_types": layer_types,
    }
    if nan_skip_store:
        result["nan_skips"] = nan_skip_store
    if permutations:
        result["permuquant"] = permutations
    if hessian_diag_store:
        result["hessian_diag"] = hessian_diag_store
    if mmap_dir is not None:
        result["_mmap_temp_dir"] = mmap_dir
    return result


def _finalize_store(
    store: Dict,
    collectors: List[ActivationStatsCollector],
    targets: List[tuple],
    shapes: Dict,
    layer_types: Dict,
    hessian_block_size: int,
    collect_amax: bool,
    rot_size: int,
    permuquant: bool,
    piso: bool,
    model_name: str,
    num_samples: int,
    num_steps: int,
    seed: int,
    latent_shape: tuple,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
    mmap_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    hessian_format: str = "block",
    dlr_rank: int = 0,
) -> Dict:
    """Post-process a store after sampling: normalise, validate, and build the result dict.

    **Mutates** *store* (pops row counts and mu2).  Do not reuse the store
    after calling this function.

    Phase ordering is documented at the module level above the helper
    definitions and must be preserved.
    """
    # Phase 1: NaN-skip report
    nan_skip_store, total_nan_skips, layers_with_skips = _log_nan_skip_report(collectors)

    # Phase 2: Normalise Hessians by row count
    _normalize_hessians_by_row_count(store)

    # Phase 3: PermuQuant permutations (pops mu2, mutates hessians)
    permutations: Dict[str, torch.Tensor] = {}
    if permuquant:
        permutations = _compute_permuquant_permutations(store, shapes, layer_types)
        _apply_permuquant_to_hessians(store, permutations)

    # Phase 4: Validate Hessians (read-only)
    _validate_hessians(store.get("hessians", {}))

    # Phase 5: PiSO diagonal extraction (read-only)
    hessian_diag_store: Dict[str, torch.Tensor] = {}
    if piso:
        hessian_diag_store = _extract_piso_diagonals(store)

    # Phase 6: Crop rotation padding
    _crop_rotation_padding(store, targets, rot_size)

    # Phase 7: Build result dict
    metadata = _build_result_metadata(
        model_name=model_name, num_samples=num_samples, num_steps=num_steps,
        seed=seed, hessian_block_size=hessian_block_size,
        hessian_format=hessian_format, dlr_rank=dlr_rank,
        collect_amax=collect_amax, rot_size=rot_size,
        permuquant=permuquant, piso=piso,
        hessian_diag_store=hessian_diag_store,
        sigma_min=sigma_min, sigma_max=sigma_max,
        targets=targets, latent_shape=latent_shape,
        layers_with_skips=layers_with_skips, total_nan_skips=total_nan_skips,
    )

    return _assemble_result(
        store=store, metadata=metadata, shapes=shapes, layer_types=layer_types,
        collect_amax=collect_amax, nan_skip_store=nan_skip_store,
        permutations=permutations, hessian_diag_store=hessian_diag_store,
        mmap_dir=mmap_dir,
    )


def collect_stats(model_patcher,
                  conditioning,
                  num_steps: int = 4,
                  num_samples: int = 16,
                  seed: int = 0,
                  latent_height: int = 64,
                  latent_width: int = 64,
                  hessian_block_size: int = 128,
                  collect_amax: bool = True,
                  rot_size: int = 0,
                  output_path: Optional[str] = None,
                  progress_callback: Optional[Callable[[int, int, str], None]] = None,
                  permuquant: bool = False,
                  piso: bool = False,
                  sigma_min: float = 0.0,
                  sigma_max: float = 1.0,
                  hessian_format: str = "block",
                  dlr_rank: int = 0,
                  force_cpu_hook: bool = False) -> Dict:
    """Run partial denoising and collect per-layer activation statistics.

    Returns a dict with keys: ``metadata``, ``hessians``, ``amax``,
    ``shapes``, ``layer_types``, and optionally ``permuquant``,
    ``hessian_diag``, ``nan_skips``.

    Side effects: writes a ``.pt`` file if *output_path* is given;
    creates and cleans up a ``.gptq_hessian_tmp/`` directory for mmap mode.
    No weights are modified.
    """
    if model_patcher is None:
        raise ValueError("model_patcher is required")
    if not conditioning:
        raise ValueError("conditioning is required (must be non-empty)")

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    # Early validation before allocation; also re-validated inside _build_sigmas.
    if sigma_min != 0.0 or sigma_max != 1.0:
        _validate_sigma_clip(sigma_min, sigma_max)
    if hessian_format == "dlr" and dlr_rank < 1:
        raise ValueError("dlr_rank must be >= 1 when hessian_format='dlr'")

    store: Dict = {
        "hessians": {},
    }
    if collect_amax:
        store["amax"] = {}

    use_mmap_full = hessian_format == "full" or (hessian_block_size == 0 and hessian_format != "dlr")
    mmap_dir: Optional[str] = None
    if use_mmap_full:
        mmap_dir = _resolve_mmap_dir(output_path)
        logger.info("Memory-mapping full Hessians to disk under %s", mmap_dir)

    t_alloc = time.time()
    collectors, targets, shapes, layer_types = _allocate_layer_buffers(
        model_patcher, store, hessian_block_size, collect_amax, rot_size,
        permuquant, mmap_dir, use_mmap_full, alloc_start=t_alloc,
        hessian_format=hessian_format, dlr_rank=dlr_rank,
        device=str(model_patcher.load_device), force_cpu=force_cpu_hook,
    )

    _run_samples(model_patcher, conditioning, num_steps, num_samples, seed,
                 latent_height, latent_width, progress_callback, pass_label="stats",
                 sigma_min=sigma_min, sigma_max=sigma_max)

    for c in collectors:
        c.remove()

    return _finalize_store(
        store=store, collectors=collectors, targets=targets, shapes=shapes,
        layer_types=layer_types, hessian_block_size=hessian_block_size,
        collect_amax=collect_amax, rot_size=rot_size, permuquant=permuquant,
        piso=piso, model_name=_guess_model_name(model_patcher),
        num_samples=num_samples, num_steps=num_steps, seed=seed,
        latent_shape=_infer_latent_shape(model_patcher, latent_height, latent_width),
        sigma_min=sigma_min, sigma_max=sigma_max, mmap_dir=mmap_dir,
        hessian_format=hessian_format, dlr_rank=dlr_rank,
    )


# ── Dual-model calibration ─────────────────────────────────────────────────


class _DualModelGuider(comfy.samplers.CFGGuider):
    """Calibration-only CFG guider routing positive/negative passes to separate models.

    Private attributes set during :meth:`outer_sample`:

    - ``uncond_inner``: the loaded negative-model inner guider (or None at CFG=1).
    - ``uncond_loaded``: list of models loaded for the negative pass.
    - ``_uncond_neg``: processed negative conditioning list.
    - ``_uncond_conds``: resolved negative conditions for :meth:`predict_noise`.
    """

    def __init__(self, model_patcher, uncond_model_patcher):
        super().__init__(model_patcher)
        self.uncond_model_patcher = uncond_model_patcher
        self.uncond_inner = None

    def outer_sample(self, noise, latent_image, sampler, sigmas,
                     denoise_mask=None, callback=None, disable_pbar=False,
                     seed=None, latent_shapes=None):
        self.uncond_inner = None
        self.uncond_loaded = []
        self._uncond_neg = None
        if not math.isclose(self.cfg, 1.0):
            uc = {"negative": list(map(lambda a: a.copy(), self.conds["negative"]))}
            self.uncond_inner, uc, self.uncond_loaded = comfy.sampler_helpers.prepare_sampling(
                self.uncond_model_patcher, noise.shape, uc,
                self.uncond_model_patcher.model_options,
            )
            self._uncond_neg = uc["negative"]
            self.uncond_model_patcher.pre_run()
        try:
            return super().outer_sample(
                noise, latent_image, sampler, sigmas, denoise_mask, callback,
                disable_pbar, seed, latent_shapes=latent_shapes,
            )
        finally:
            if self.uncond_inner is not None:
                self.uncond_model_patcher.cleanup()
                comfy.sampler_helpers.cleanup_models(
                    {"negative": self._uncond_neg}, self.uncond_loaded,
                )
                self.uncond_inner = None

    def inner_sample(self, noise, latent_image, device, sampler, sigmas,
                     denoise_mask, callback, disable_pbar, seed,
                     latent_shapes=None):
        if self.uncond_inner is not None:
            li = latent_image
            if li is not None and torch.count_nonzero(li) > 0:
                li = self.uncond_inner.process_latent_in(li)
            self._uncond_conds = comfy.samplers.process_conds(
                self.uncond_inner, noise, {"negative": self._uncond_neg}, device,
                li, denoise_mask, seed, latent_shapes=latent_shapes,
            )["negative"]
        return super().inner_sample(
            noise, latent_image, device, sampler, sigmas, denoise_mask,
            callback, disable_pbar, seed, latent_shapes=latent_shapes,
        )

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        cond = comfy.samplers.calc_cond_batch(
            self.inner_model, [positive], x, timestep, model_options,
        )[0]
        if self.uncond_inner is None or (
            math.isclose(self.cfg, 1.0)
            and not model_options.get("disable_cfg1_optimization", False)
        ):
            return cond

        uncond_model_options = model_options
        if "multigpu_clones" in model_options:
            uncond_model_options = {
                k: v for k, v in model_options.items() if k != "multigpu_clones"
            }
        uncond = comfy.samplers.calc_cond_batch(
            self.uncond_inner, [self._uncond_conds], x, timestep,
            uncond_model_options,
        )[0]
        return comfy.samplers.cfg_function(
            self.inner_model, cond, uncond, self.cfg, x, timestep,
            model_options=model_options, cond=positive, uncond=self._uncond_conds,
        )


def collect_stats_dual(model_patcher,
                       model_negative_patcher,
                       positive,
                       negative,
                       cfg: float = 4.0,
                       num_steps: int = 4,
                       num_samples: int = 16,
                       seed: int = 0,
                       latent_height: int = 64,
                       latent_width: int = 64,
                       hessian_block_size: int = 128,
                       collect_amax: bool = True,
                       rot_size: int = 0,
                       output_path_positive: Optional[str] = None,
                       output_path_negative: Optional[str] = None,
                       progress_callback: Optional[Callable[[int, int, str], None]] = None,
                       permuquant: bool = False,
                       piso: bool = False,
                       hessian_format: str = "block",
                       dlr_rank: int = 0,
                       sigma_min: float = 0.0,
                       sigma_max: float = 1.0,
                       force_cpu_hook: bool = False) -> Tuple[Dict, Dict]:
    """Calibrate two models used together via dual-model CFG (e.g. Ideogram 4).

    Returns ``(result_positive, result_negative)`` in the same schema as
    ``collect_stats``.  Raises ``ValueError`` when *cfg == 1.0*.

    Side effects: writes two ``.pt`` files; creates and cleans up
    ``.gptq_hessian_tmp/`` directories for mmap mode.  No weights modified.
    """
    if model_patcher is None or model_negative_patcher is None:
        raise ValueError("Both model and model_negative are required")
    if positive is None:
        raise ValueError("positive conditioning is required")
    if math.isclose(cfg, 1.0):
        raise ValueError(
            "cfg=1.0 disables the negative pass — the negative model will "
            "not be calibrated.  Use cfg > 1.0 (e.g., 4.0) to calibrate both."
        )

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if hessian_format == "dlr" and dlr_rank < 1:
        raise ValueError("dlr_rank must be >= 1 when hessian_format='dlr'")
    if sigma_min != 0.0 or sigma_max != 1.0:
        _validate_sigma_clip(sigma_min, sigma_max)

    # ── Allocate stores for both models ──────────────────────────────
    store_pos: Dict = {"hessians": {}}
    store_neg: Dict = {"hessians": {}}
    if collect_amax:
        store_pos["amax"] = {}
        store_neg["amax"] = {}

    use_mmap_full = hessian_format == "full" or (hessian_block_size == 0 and hessian_format != "dlr")
    mmap_dir_pos = _resolve_mmap_dir(output_path_positive) if use_mmap_full else None
    mmap_dir_neg = _resolve_mmap_dir(output_path_negative) if use_mmap_full else None

    t_alloc = time.time()
    collectors_pos, targets_pos, shapes_pos, types_pos = _allocate_layer_buffers(
        model_patcher, store_pos, hessian_block_size, collect_amax, rot_size,
        permuquant, mmap_dir_pos, use_mmap_full, alloc_start=t_alloc,
        hessian_format=hessian_format, dlr_rank=dlr_rank,
        device=str(model_patcher.load_device), force_cpu=force_cpu_hook,
    )
    collectors_neg, targets_neg, shapes_neg, types_neg = _allocate_layer_buffers(
        model_negative_patcher, store_neg, hessian_block_size, collect_amax,
        rot_size, permuquant, mmap_dir_neg, use_mmap_full, alloc_start=t_alloc,
        hessian_format=hessian_format, dlr_rank=dlr_rank,
        device=str(model_negative_patcher.load_device), force_cpu=force_cpu_hook,
    )

    logger.info(
        "Dual calibration: %d + %d layers, %d samples, %d steps, cfg=%.1f",
        len(targets_pos), len(targets_neg), num_samples, num_steps, cfg,
    )

    # ── Set up the dual guider and run sampling ──────────────────────
    guider = _DualModelGuider(model_patcher, model_negative_patcher)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)

    sampler = comfy.samplers.sampler_object(_DEFAULT_SAMPLER)
    sigmas = _build_sigmas(_resolve_model_sampling(model_patcher), num_steps,
                           sigma_min=sigma_min, sigma_max=sigma_max)
    sigmas = sigmas.to(model_patcher.load_device)

    fallback_pbar = comfy.utils.ProgressBar(num_samples) if progress_callback is None else None
    overall_start = time.time()

    for sample_idx in range(max(1, num_samples)):
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + sample_idx)
        channels, lat_h, lat_w = _infer_latent_shape(model_patcher, latent_height, latent_width)
        noise = torch.randn(
            (1, channels, lat_h, lat_w), generator=generator, dtype=torch.float32,
        )
        sample_start = time.time()
        try:
            guider.sample(
                noise=noise, latent_image=noise, sampler=sampler,
                sigmas=sigmas, disable_pbar=True,
            )
        except Exception as exc:
            logger.exception("Dual sample %d/%d failed: %s", sample_idx + 1, num_samples, exc)
            raise

        done = sample_idx + 1
        elapsed, avg, remaining = _compute_eta(done, num_samples, sample_start, overall_start)
        msg = f"sample {done}/{num_samples} ({_fmt_duration(elapsed)}, avg {_fmt_duration(avg)}, ETA {_fmt_duration(remaining)})"
        if progress_callback is not None:
            try:
                progress_callback(done, num_samples, msg)
            except Exception:
                pass
        if fallback_pbar is not None:
            fallback_pbar.update_absolute(done, num_samples)
        logger.info("Dual calibration: %s", msg)

        if (done % _CACHE_CLEAR_INTERVAL) == 0 and torch.cuda.is_available():
            comfy.model_management.soft_empty_cache()

    total_elapsed = time.time() - overall_start
    logger.info("Dual calibration complete: %d samples in %.1fs", num_samples, total_elapsed)

    # ── Finalize both stores ─────────────────────────────────────────
    for c in collectors_pos:
        c.remove()
    for c in collectors_neg:
        c.remove()

    latent_shape = _infer_latent_shape(model_patcher, latent_height, latent_width)

    result_pos = _finalize_store(
        store=store_pos, collectors=collectors_pos, targets=targets_pos,
        shapes=shapes_pos, layer_types=types_pos,
        hessian_block_size=hessian_block_size, collect_amax=collect_amax,
        rot_size=rot_size, permuquant=permuquant, piso=piso,
        model_name=_guess_model_name(model_patcher),
        num_samples=num_samples, num_steps=num_steps, seed=seed,
        latent_shape=latent_shape, mmap_dir=mmap_dir_pos,
        hessian_format=hessian_format, dlr_rank=dlr_rank,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )
    result_neg = _finalize_store(
        store=store_neg, collectors=collectors_neg, targets=targets_neg,
        shapes=shapes_neg, layer_types=types_neg,
        hessian_block_size=hessian_block_size, collect_amax=collect_amax,
        rot_size=rot_size, permuquant=permuquant, piso=piso,
        model_name=_guess_model_name(model_negative_patcher),
        num_samples=num_samples, num_steps=num_steps, seed=seed,
        latent_shape=latent_shape, mmap_dir=mmap_dir_neg,
        hessian_format=hessian_format, dlr_rank=dlr_rank,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )

    result_pos["metadata"]["dual_model"] = True
    result_pos["metadata"]["dual_role"] = "positive"
    result_pos["metadata"]["dual_cfg"] = float(cfg)
    result_neg["metadata"]["dual_model"] = True
    result_neg["metadata"]["dual_role"] = "negative"
    result_neg["metadata"]["dual_cfg"] = float(cfg)

    return result_pos, result_neg


def _guess_model_name(model_patcher) -> str:
    """Return a human-readable model name derived from the patcher's class hierarchy."""
    inner = getattr(model_patcher, "model", None)
    if inner is None:
        return "unknown"
    cls = inner.__class__.__name__
    config = getattr(inner, "model_config", None)
    if config is not None:
        return f"{cls}:{config.__class__.__name__}"
    return cls
