"""Per-layer activation statistics collection for GPTQ calibration.

Two phases: (1) register forward_pre_hooks that accumulate ``H += xᵀx``
and optionally ``amax``, (2) post-process: normalise, validate, crop padding.

Glossary: Hessian = unnormalised Gram matrix; amax = running max |x|;
mu2 = per-channel second moment; rot_size = Hadamard group size;
block_size = on-diagonal Hessian block size (0 = full).
"""

import datetime
import gc
import logging
import math
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.sampler_helpers

import numpy as np
import os
import tempfile


logger = logging.getLogger("comfyui_gptq_calibration")


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


# Optional: Triton FHT kernel for O(N log N) rotation on GPU.
# Falls back to the CPU matmul path when Triton/CUDA unavailable.
try:
    from kernels.triton_fht_rotate import fht_rotate as _fht_rotate
except ImportError:
    _fht_rotate = None


# ── ConvRot Hadamard rotation ──

# Hadamard construction is O(n²) via Kronecker product — caching avoids
# redundant reconstruction when the same rot_size is used across layers.
_HADAMARD_CACHE: Dict[tuple, torch.Tensor] = {}

def _is_power_of_four(n: int) -> bool:
    """Return True if n is a power of 4 (4, 16, 64, 256, ...)."""
    if n < 4:
        return False
    return (n & (n - 1)) == 0 and (n & 0x55555555) == n

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


def permute_hessian(H, perm: torch.Tensor):
    """Permute a Hessian matrix according to a channel reordering.

    Supports: ``dict`` (DLR), ``list[Tensor]`` (block-diagonal),
    ``Tensor(dim=3)`` (stacked blocks), ``Tensor(dim=2)`` (full).
    """
    perm = perm.to(torch.int64)

    if isinstance(H, dict) and H.get("format") == "dlr":
        perm = perm.to(H["D"].device)
        new_H = dict(H)
        new_H["D"] = H["D"][perm]
        new_H["U"] = H["U"][perm]
        return new_H

    if isinstance(H, list):
        # List processing builds CPU helper tensors — move perm to CPU.
        perm = perm.cpu()
        block_sizes = [b.shape[0] for b in H]
        num_blocks = len(block_sizes)
        offsets = [0]
        for bs in block_sizes:
            offsets.append(offsets[-1] + bs)
        n = offsets[-1]

        # Phase 1: build global_index → (block, local_index) lookup
        block_of = torch.empty(n, dtype=torch.int64)
        local_of = torch.empty(n, dtype=torch.int64)
        for bi, bs in enumerate(block_sizes):
            s = offsets[bi]
            block_of[s:s + bs] = bi
            local_of[s:s + bs] = torch.arange(bs)

        perm_block = block_of[perm]
        perm_local = local_of[perm]

        # Phase 2: build each output block from entries whose row and col
        # originate in the same source block (cross-block entries are zero)
        new_blocks = []
        for i in range(num_blocks):
            ri = offsets[i]
            bs_i = block_sizes[i]
            rb = perm_block[ri:ri + bs_i]
            rl = perm_local[ri:ri + bs_i]

            block = torch.zeros(bs_i, bs_i, dtype=H[0].dtype)
            for li in range(bs_i):
                k = rb[li].item()
                same = (rb == k)
                if same.any():
                    block[li, same] = H[k][rl[li], rl[same]]
            new_blocks.append(block)

        del H
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
            del H
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
    An exact per-channel diagonal ``_diag`` is accumulated separately
    and used to form the residual diagonal in :meth:`dlr_decompose`.
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

    def dlr_decompose(self) -> tuple:
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
        handle = module.register_forward_pre_hook(self._hook_fn)
        self.hooks.append(handle)

    def remove(self) -> None:
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
        """Block-diagonal only when n_features is much larger than block_size
        (×4 threshold avoids accuracy loss on small layers).
        """
        return self.hessian_block_size > 0 and n_features > self.hessian_block_size * 4

    def _make_fd(self, n_features: int) -> FrequentDirections:
        """Create a FrequentDirections sketch sized for this layer."""
        rank = min(self.dlr_rank, n_features)
        sketch_size = 2 * rank + 4
        return FrequentDirections(n=n_features, sketch_size=sketch_size, rank=rank)

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
        """Accumulate ``xᵀx`` into the Hessian for this layer."""
        hessians = self.store.setdefault("hessians", {})
        row_counts = self.store.setdefault("_hessian_row_count", {})
        n_rows = x_flat.shape[0]

        existing = hessians.get(self.layer_name)
        if existing is None:
            if self.hessian_format == "dlr":
                fd = self._make_fd(x_flat.shape[1])
                fd.update(x_flat)
                hessians[self.layer_name] = fd
            elif self._use_block_gram(x_flat.shape[1]):
                hessians[self.layer_name] = self._block_gram(x_flat, self.hessian_block_size)
            else:
                hessians[self.layer_name] = x_flat.T @ x_flat
        else:
            if isinstance(existing, FrequentDirections):
                existing.update(x_flat)
            elif isinstance(existing, list):
                block_size = self.hessian_block_size
                for i, block in enumerate(existing):
                    start = i * block_size
                    end = min((i + 1) * block_size, x_flat.shape[1])
                    if start >= x_flat.shape[1]:
                        break
                    xi = x_flat[:, start:end]
                    block.add_(xi.T @ xi)
            elif existing.dim() == 3:
                # backward-compat for old stacked-tensor format
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
            else:
                # Full Hessian (Tensor, possibly mmap'd on CPU).
                # When force_cpu=False the activation is on GPU but the mmap
                # buffer is on CPU — move the activation to match.
                if x_flat.device != existing.device:
                    x_flat = x_flat.to(existing.device)
                existing.add_(x_flat.T @ x_flat)

        row_counts[self.layer_name] = row_counts.get(self.layer_name, 0) + n_rows

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

    def _hook_fn(self, module: nn.Module, inputs) -> None:
        # Pipeline: detach → flatten → float32 → NaN guard → rotate → accumulate.
        # GPU-fast by default; force_cpu=True moves everything to CPU.
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
        _ACT_AMAX_CEILING = 1e6
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


def _walk_target_modules(model: nn.Module) -> List[tuple]:
    """Yield (name, module) for every Linear/Conv2d in ``model``."""
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            targets.append((name, module))
    return targets


def _get_in_features(module: nn.Module, ltype: str) -> int:
    """Return the flattened input feature count for a target layer."""
    if ltype == "Conv2d":
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


def _infer_latent_shape(model_patcher, height: int = 64, width: int = 64) -> tuple:
    """Return ``(channels, H, W)`` for the calibration noise tensor."""
    inner = getattr(model_patcher, "model", None)
    if inner is None:
        return (4, int(height), int(width))
    latent_format = getattr(inner, "latent_format", None)
    channels = getattr(latent_format, "latent_channels", 4) if latent_format is not None else 4
    return (int(channels), int(height), int(width))


def _resolve_model_sampling(model_patcher):
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
    """Check Hessians for corruption (extreme outliers, asymmetry). Logs warnings only."""
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

            # Off-diagonal > 1000× diagonal → outlier corruption
            if max_val > 1e6 and diag_max > 0 and max_val / diag_max > 1e3:
                corrupted.append(name)
                logger.warning(
                    "Hessian corruption detected in %s: max=%.0f, diag_max=%.0f, ratio=%.0f",
                    name, max_val, diag_max, max_val / diag_max,
                )
            elif sym_err > max_val * 0.01 and max_val > 1e3:
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
                if sym_err > max_val * 0.01 and max_val > 1e3:
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


def _run_samples(model_patcher, conditioning, num_steps, num_samples, seed,
                 latent_height, latent_width, progress_callback, pass_label="",
                 sigma_min=0.0, sigma_max=1.0):
    """Run the calibration sampling loop."""
    t2 = time.time()
    device = model_patcher.load_device
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    channels, lat_h, lat_w = _infer_latent_shape(model_patcher, latent_height, latent_width)
    batch = 1

    sigmas = _build_sigmas(_resolve_model_sampling(model_patcher), num_steps,
                           sigma_min=sigma_min, sigma_max=sigma_max).to(device)
    positive = conditioning
    negative = conditioning
    logger.info("model setup (%s): %.2fs", pass_label, time.time() - t2)

    total = max(1, int(num_samples))
    overall_start = time.time()
    for sample_idx in range(total):
        sample_start = time.time()
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
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
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

        elapsed = time.time() - sample_start
        done = sample_idx + 1
        total_elapsed = time.time() - overall_start
        avg = total_elapsed / done
        remaining = avg * (total - done)
        msg = f"sample {done}/{total} ({_fmt_duration(elapsed)}, avg {_fmt_duration(avg)}, ETA {_fmt_duration(remaining)})"
        if progress_callback is not None:
            try:
                progress_callback(done, total, msg)
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)

        # Periodically clear CUDA cache to prevent OOM during long runs
        if (done % 4) == 0 and torch.cuda.is_available():
            comfy.model_management.soft_empty_cache()

    total_elapsed = time.time() - overall_start
    logger.info("Calibration (%s) complete: %d samples in %.1fs", pass_label, total, total_elapsed)


def _extract_hessian_diag(H) -> Optional[torch.Tensor]:
    """Extract the diagonal from a Hessian (full, block-diagonal, or DLR)."""
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




# ── Refactored helpers shared by collect_stats and collect_stats_dual ───────


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
) -> tuple:
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
        ltype = "Conv2d" if isinstance(module, nn.Conv2d) else "Linear"
        layer_types[name] = ltype
        shapes[name] = tuple(module.weight.shape)

        in_f = _get_in_features(module, ltype)
        alloc_f = in_f
        if rot_size > 0 and in_f % rot_size != 0:
            alloc_f = rot_size * ((in_f + rot_size - 1) // rot_size)

        if hessian_format == "dlr" and dlr_rank > 0:
            rank = min(dlr_rank, alloc_f)
            sketch_size = 2 * rank + 4
            store["hessians"][name] = FrequentDirections(
                n=alloc_f, sketch_size=sketch_size, rank=rank,
                device=hessian_device,
            )
        elif hessian_block_size > 0 and alloc_f > hessian_block_size * 4:
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
    """
    # ── NaN-skip report ──────────────────────────────────────────────
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

    # ── Normalise Hessians by row count ──────────────────────────────
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

    # ── PermuQuant: compute permutations and apply to Hessians ──
    permutations: Dict[str, torch.Tensor] = {}
    if permuquant:
        mu2_store = store.pop("mu2", {})
        count_store = store.pop("_mu2_count", {})
        for name, mu2_sum in mu2_store.items():
            n = count_store.get(name, 1)
            mu2 = mu2_sum / n
            shape = shapes[name]
            ltype = layer_types[name]
            if ltype == "Conv2d":
                in_f = shape[1] * shape[2] * shape[3]
            else:
                in_f = shape[1]
            mu2 = mu2[:in_f]
            perm = mu2.argsort(descending=True).to(torch.int32)
            permutations[name] = perm
        logger.info("PermuQuant: computed permutations for %d layers", len(permutations))

        hessians = store.get("hessians", {})
        for name, perm in permutations.items():
            if name in hessians:
                old = hessians[name]
                hessians[name] = permute_hessian(old, perm)
                del old
        gc.collect()
        logger.info("PermuQuant: applied permutations to Hessians")

    # ── Validate Hessians ────────────────────────────────────────────
    _validate_hessians(store.get("hessians", {}))

    # ── PiSO: extract Hessian diagonal ───────────────────────────────
    hessian_diag_store: Dict[str, torch.Tensor] = {}
    if piso:
        hessians = store.get("hessians", {})
        for name, H in hessians.items():
            diag = _extract_hessian_diag(H)
            if diag is not None:
                hessian_diag_store[name] = diag
        logger.info("PiSO: extracted Hessian diagonal for %d layers", len(hessian_diag_store))

    # ── Crop rotation padding ────────────────────────────────────────
    if rot_size > 0:
        for name, module in targets:
            ltype = "Conv2d" if isinstance(module, nn.Conv2d) else "Linear"
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

    # ── Build result dict ────────────────────────────────────────────
    channels, lat_h, lat_w = latent_shape

    metadata = {
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
        "recommended_damping_ratio": 0.01,
        "quantization_order": "column",
        "num_layers": len(targets),
        "latent_shape": [int(channels), int(lat_h), int(lat_w)],
        "nan_skip_layers": layers_with_skips,
        "nan_skip_total": total_nan_skips,
    }

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
    ``shapes``, ``layer_types``, and optionally ``permuquant``.
    """
    if model_patcher is None:
        raise ValueError("model_patcher is required")
    if not conditioning:
        raise ValueError("conditioning is required (must be non-empty)")

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if sigma_min != 0.0 or sigma_max != 1.0:
        _validate_sigma_clip(sigma_min, sigma_max)
    if hessian_format == "dlr" and dlr_rank < 1:
        raise ValueError("dlr_rank must be >= 1 when hessian_format='dlr'")

    store: Dict = {
        "hessians": {},
    }
    if collect_amax:
        store["amax"] = {}

    use_mmap_full = (hessian_block_size == 0 and hessian_format != "dlr")
    mmap_dir = None
    if use_mmap_full:
        if output_path:
            base_dir = os.path.dirname(os.path.abspath(output_path))
            mmap_dir = os.path.join(base_dir, ".gptq_hessian_tmp")
        else:
            mmap_dir = os.path.join(tempfile.gettempdir(), "gptq_hessian_tmp")
        os.makedirs(mmap_dir, exist_ok=True)
        logger.info("Memory-mapping full Hessians to disk under %s", mmap_dir)

    collectors, targets, shapes, layer_types = _allocate_layer_buffers(
        model_patcher, store, hessian_block_size, collect_amax, rot_size,
        permuquant, mmap_dir, use_mmap_full,
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
    """Calibration-only CFG guider routing positive/negative passes to separate models."""

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
                       force_cpu_hook: bool = False) -> tuple:
    """Calibrate two models used together via dual-model CFG (e.g. Ideogram 4).

    Returns ``(result_positive, result_negative)`` in the same schema as
    ``collect_stats``.  Raises ``ValueError`` when *cfg == 1.0*.
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

    # ── Allocate stores for both models ──────────────────────────────
    store_pos: Dict = {"hessians": {}}
    store_neg: Dict = {"hessians": {}}
    if collect_amax:
        store_pos["amax"] = {}
        store_neg["amax"] = {}

    use_mmap_full = (hessian_block_size == 0 and hessian_format != "dlr")

    def _resolve_mmap(base_path: Optional[str]) -> Optional[str]:
        if not use_mmap_full:
            return None
        if base_path:
            d = os.path.join(os.path.dirname(os.path.abspath(base_path)), ".gptq_hessian_tmp")
        else:
            d = os.path.join(tempfile.gettempdir(), "gptq_hessian_tmp")
        os.makedirs(d, exist_ok=True)
        return d

    mmap_dir_pos = _resolve_mmap(output_path_positive)
    mmap_dir_neg = _resolve_mmap(output_path_negative)

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

    sampler = comfy.samplers.sampler_object("euler")
    sigmas = _build_sigmas(_resolve_model_sampling(model_patcher), num_steps)
    sigmas = sigmas.to(model_patcher.load_device)

    progress = comfy.utils.ProgressBar(num_samples) if progress_callback is None else None
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

        elapsed = time.time() - sample_start
        done = sample_idx + 1
        total_elapsed = time.time() - overall_start
        avg = total_elapsed / done
        remaining = avg * (num_samples - done)
        msg = f"sample {done}/{num_samples} ({_fmt_duration(elapsed)}, avg {_fmt_duration(avg)}, ETA {_fmt_duration(remaining)})"
        if progress_callback is not None:
            try:
                progress_callback(done, num_samples, msg)
            except Exception:
                pass
        if progress is not None:
            progress.update_absolute(done, num_samples)
        logger.info("Dual calibration: %s", msg)

        if (done % 4) == 0 and torch.cuda.is_available():
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
    )

    result_pos["metadata"]["dual_model"] = True
    result_pos["metadata"]["dual_role"] = "positive"
    result_pos["metadata"]["dual_cfg"] = float(cfg)
    result_neg["metadata"]["dual_model"] = True
    result_neg["metadata"]["dual_role"] = "negative"
    result_neg["metadata"]["dual_cfg"] = float(cfg)

    return result_pos, result_neg


def _guess_model_name(model_patcher) -> str:
    inner = getattr(model_patcher, "model", None)
    if inner is None:
        return "unknown"
    cls = inner.__class__.__name__
    config = getattr(inner, "model_config", None)
    if config is not None:
        return f"{cls}:{config.__class__.__name__}"
    return cls
