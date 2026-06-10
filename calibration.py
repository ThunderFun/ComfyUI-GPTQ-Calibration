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

import numpy as np
import os
import tempfile


logger = logging.getLogger("comfyui_gptq_calibration")


# ── ConvRot Hadamard rotation ──

_HADAMARD_CACHE: Dict[tuple, torch.Tensor] = {}

def _is_power_of_four(n: int) -> bool:
    """Return True if n is a power of 4 (4, 16, 64, 256, ...)."""
    if n < 4:
        return False
    return (n & (n - 1)) == 0 and (n & 0x55555555) == n

def get_hadamard(size: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Return a normalized Hadamard matrix of the given size.

    Uses Regular Hadamard for powers of 4 (ConvRot paper: balanced row
    sums prevent row-wise outlier aggregation). Falls back to Sylvester
    construction for other powers of 2.

    The matrix is cached after first construction. ``size`` must be a
    power of 2.
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
    """Apply group-wise Hadamard rotation to activations.

    Pads the last dimension to a multiple of ``rot_size`` if needed,
    then applies the normalized Hadamard transform independently to
    each group. Used for ConvRot-style Hessian rotation.
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
    """Apply a channel permutation to a Hessian matrix (full or block-diagonal).

    For full 2D Hessians: H_perm = H[perm][:, perm].
    For block-diagonal Hessians: builds each output block directly from
    the source diagonal blocks without materializing the full n*n matrix.
    Block-diagonal list format is preserved.

    Args:
        H: Hessian tensor (2D or 3D) or list of block tensors
        perm: [in_features] permutation indices (int32 or int64)

    Returns:
        Permuted Hessian in the same format as the input
    """
    perm = perm.to(torch.int64)

    if isinstance(H, list):
        block_sizes = [b.shape[0] for b in H]
        num_blocks = len(block_sizes)
        offsets = [0]
        for bs in block_sizes:
            offsets.append(offsets[-1] + bs)
        n = offsets[-1]

        # Precompute: global index → (source block, local index)
        block_of = torch.empty(n, dtype=torch.int64)
        local_of = torch.empty(n, dtype=torch.int64)
        for bi, bs in enumerate(block_sizes):
            s = offsets[bi]
            block_of[s:s + bs] = bi
            local_of[s:s + bs] = torch.arange(bs)

        perm_block = block_of[perm]
        perm_local = local_of[perm]

        # Build each diagonal output block directly — never create n*n matrix
        new_blocks = []
        for i in range(num_blocks):
            ri = offsets[i]
            bs_i = block_sizes[i]
            rb = perm_block[ri:ri + bs_i]
            rl = perm_local[ri:ri + bs_i]

            # Diagonal block: only entries where row and col map to same source block
            block = torch.zeros(bs_i, bs_i, dtype=H[0].dtype)
            for li in range(bs_i):
                k = rb[li].item()
                # Which cols in this diagonal block share the same source block?
                same = (rb == k)
                if same.any():
                    block[li, same] = H[k][rl[li], rl[same]]
            new_blocks.append(block)

        del H
        return new_blocks

    if isinstance(H, torch.Tensor):
        if H.dim() == 2:
            return H[perm][:, perm]
        if H.dim() == 3:
            block_size = H.shape[1]
            num_blocks = H.shape[0]
            n = num_blocks * block_size
            H_full = torch.zeros(n, n, dtype=H.dtype)
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


class ActivationStatsCollector:
    """Forward pre-hook that accumulates activation statistics for one layer.

    For each forward pass the hook flattens the layer's input to 2D
    (using ``nn.Unfold`` for ``nn.Conv2d``) and adds
    ``x.T @ x`` to a per-layer Hessian on CPU, and tracks the running
    maximum of ``abs(x)`` (amax).

    The Hessian can be stored either as the full ``(in_features,
    in_features)`` matrix (default, paper-accurate) or as a stack of
    diagonal blocks of size ``hessian_block_size`` to save disk space.

    When ``mode="mu2"``, only per-channel second moments are accumulated
    (for PermuQuant channel reordering). No Hessian is stored.
    When ``mode="both"``, both mu2 and Hessian are accumulated in a single
    pass. The caller can then compute permutations from mu2 and apply them
    to the collected Hessians after the fact.
    When ``permutation`` is provided, activations are reindexed before
    Hessian accumulation.
    """

    def __init__(self,
                 layer_name: str,
                 store: Dict,
                 layer_type: str,
                 hessian_block_size: int = 0,
                 collect_amax: bool = True,
                 rot_size: int = 0,
                 mode: str = "hessian",
                 permutation: Optional[torch.Tensor] = None):
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.store = store
        self.hessian_block_size = int(hessian_block_size)
        self.collect_amax = bool(collect_amax)
        self.rot_size = int(rot_size)
        self.mode = mode
        self.permutation = permutation
        self.hooks: List[torch.utils.hooks.RemovableHook] = []

    def register(self, module: nn.Module) -> None:
        handle = module.register_forward_pre_hook(self._hook_fn)
        self.hooks.append(handle)

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    # ---- internal helpers ------------------------------------------------

    def _use_block_gram(self, n_features: int) -> bool:
        return self.hessian_block_size > 0 and n_features > self.hessian_block_size * 4

    def _flatten_linear_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-1])

    def _flatten_conv_input(self, x: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
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
        hessians = self.store.setdefault("hessians", {})
        existing = hessians.get(self.layer_name)
        if existing is None:
            if self._use_block_gram(x_flat.shape[1]):
                hessians[self.layer_name] = self._block_gram(x_flat, self.hessian_block_size)
            else:
                hessians[self.layer_name] = x_flat.T @ x_flat
        else:
            if isinstance(existing, list):
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
                existing.add_(x_flat.T @ x_flat)

    def _block_gram(self, x: torch.Tensor, block_size: int) -> List[torch.Tensor]:
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
        if not inputs:
            return
        x = inputs[0].detach()
        if isinstance(module, nn.Conv2d):
            x_flat = self._flatten_conv_input(x, module)
        else:
            x_flat = self._flatten_linear_input(x)

        if x_flat.numel() == 0:
            return

        # Move to CPU before accumulation
        x_flat = x_flat.cpu().float()

        if self.rot_size > 0:
            x_flat = rotate_activations(x_flat, self.rot_size)

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
    """Yield (qualified_name, module) for every Linear/Conv2d in ``model``.

    The traversal uses ``isinstance`` so that custom subclasses of
    ``nn.Linear`` and ``nn.Conv2d`` are also picked up.
    """
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
) -> torch.Tensor:
    """Create a zero-initialized Hessian tensor, optionally memory-mapped to disk."""
    if not use_mmap:
        return torch.zeros(shape, dtype=torch.float32)
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
    """Return ``(channels, H, W)`` for the calibration noise tensor.

    The ``height`` and ``width`` arguments are **latent** dimensions
    (i.e. the spatial size of the latent the model expects). The
    channel count is taken from the model's ``latent_format`` if
    available, else defaults to 4.
    """
    inner = getattr(model_patcher, "model", None)
    if inner is None:
        return (4, int(height), int(width))
    latent_format = getattr(inner, "latent_format", None)
    channels = getattr(latent_format, "latent_channels", 4) if latent_format is not None else 4
    return (int(channels), int(height), int(width))


def _resolve_model_sampling(model_patcher):
    inner = getattr(model_patcher, "model", None)
    return getattr(inner, "model_sampling", None) if inner is not None else None


def _build_sigmas(model_sampling, num_steps: int) -> torch.Tensor:
    """Return a tensor of ``num_steps + 1`` sigmas ending at 0.

    Uses ``simple_scheduler`` if a model_sampling is available (which gives
    us the proper sigma_min/sigma_max). Otherwise falls back to a linear
    schedule from 1.0 to 0.0 which is a reasonable approximation for
    most flow-matching / EDM-style models.
    """
    if model_sampling is not None:
        try:
            return comfy.samplers.simple_scheduler(model_sampling, num_steps)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("simple_scheduler failed (%s); falling back to linear", exc)
    return torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)


def _validate_hessians(hessians: Dict[str, torch.Tensor]) -> None:
    """Check accumulated Hessians for corruption (extreme outliers, asymmetry).

    Logs warnings for any layers with suspicious entries. Does NOT modify the data.
    """
    corrupted = []
    for name, H in hessians.items():
        if not isinstance(H, torch.Tensor):
            continue

        if H.dim() == 2:
            # Full Hessian — check symmetry and outlier elements
            sym_err = (H - H.T).abs().max().item()
            max_val = H.abs().max().item()
            diag_max = H.diagonal().abs().max().item()

            # Flag if off-diagonal element is orders of magnitude larger than diagonal
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
                 latent_height, latent_width, progress_callback, pass_label=""):
    """Run the calibration sampling loop."""
    t2 = time.time()
    device = model_patcher.load_device
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    channels, lat_h, lat_w = _infer_latent_shape(model_patcher, latent_height, latent_width)
    batch = 1

    sigmas = _build_sigmas(_resolve_model_sampling(model_patcher), num_steps).to(device)
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
        msg = f"sample {done}/{total} ({elapsed:.1f}s, avg {avg:.1f}s, ETA {remaining:.0f}s)"
        if progress_callback is not None:
            try:
                progress_callback(done, total, msg)
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)

        if (done % 4) == 0 and torch.cuda.is_available():
            comfy.model_management.soft_empty_cache()

    total_elapsed = time.time() - overall_start
    logger.info("Calibration (%s) complete: %d samples in %.1fs", pass_label, total, total_elapsed)


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
                  permuquant: bool = False) -> Dict:
    """Run partial denoising and collect per-layer activation statistics.

    Args:
        model_patcher: A ComfyUI ``ModelPatcher`` instance (already loaded).
        conditioning: A list of ``CONDITIONING`` dicts. The first batch
            element is used; duplicates are generated by re-noising.
        num_steps: Number of denoising steps per sample.
        num_samples: Number of independent samples to run.
        seed: Seed for the random noise / timestep generator.
        latent_height/latent_width: Latent spatial size to use.
        hessian_block_size: ``0`` for full H (auto memory-mapped to disk),
            else diagonal block size.
        collect_amax: Whether to track ``max(abs(x))`` per layer.
        output_path: Optional path used to place temporary mmap files when
            ``hessian_block_size == 0``.
        progress_callback: Optional ``callable(done, total, msg)``.

    Returns:
        Dict with keys ``metadata``, ``hessians``, ``amax`` (optional),
        ``shapes`` and ``layer_types``. May also contain
        ``_mmap_temp_dir`` when full-Hessian mmap is active.
    """
    if model_patcher is None:
        raise ValueError("model_patcher is required")
    if not conditioning:
        raise ValueError("conditioning is required (must be non-empty)")

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")

    t0 = time.time()
    target = _resolve_inner_model(model_patcher)
    logger.info("resolve model: %.2fs", time.time() - t0)

    t1 = time.time()
    targets = _walk_target_modules(target)
    logger.info("walk modules: %.2fs (%d target layers)", time.time() - t1, len(targets))

    store: Dict = {
        "hessians": {},
    }
    if collect_amax:
        store["amax"] = {}

    shapes: Dict = {}
    layer_types: Dict = {}

    logger.info("Collecting stats for %d layers, %d samples, %d steps each", len(targets), num_samples, num_steps)

    use_mmap_full = (hessian_block_size == 0)
    mmap_dir = None
    if use_mmap_full:
        if output_path:
            base_dir = os.path.dirname(os.path.abspath(output_path))
            mmap_dir = os.path.join(base_dir, ".gptq_hessian_tmp")
        else:
            mmap_dir = os.path.join(tempfile.gettempdir(), "gptq_hessian_tmp")
        os.makedirs(mmap_dir, exist_ok=True)
        logger.info("Memory-mapping full Hessians to disk under %s", mmap_dir)

    # ── Main pass: collect Hessians (and mu2 if permuquant) ──
    # When permuquant is enabled we collect mu2 and Hessian in a single pass
    # (mode="both") to avoid running the denoising loop twice — the second
    # pass would see corrupted model state and produce NaN activations.
    collector_mode = "both" if permuquant else "hessian"
    permutations: Dict[str, torch.Tensor] = {}
    collectors: List[ActivationStatsCollector] = []
    alloc_start = time.time()

    for idx, (name, module) in enumerate(targets, 1):
        ltype = "Conv2d" if isinstance(module, nn.Conv2d) else "Linear"
        layer_types[name] = ltype
        if ltype == "Conv2d":
            shapes[name] = tuple(module.weight.shape)
        else:
            shapes[name] = tuple(module.weight.shape)

        in_f = _get_in_features(module, ltype)
        alloc_f = in_f
        if rot_size > 0 and in_f % rot_size != 0:
            alloc_f = rot_size * ((in_f + rot_size - 1) // rot_size)
        if hessian_block_size > 0 and alloc_f > hessian_block_size * 4:
            num_blocks = (alloc_f + hessian_block_size - 1) // hessian_block_size
            blocks = []
            for i in range(num_blocks):
                start = i * hessian_block_size
                end = min((i + 1) * hessian_block_size, alloc_f)
                width = end - start
                blocks.append(torch.zeros(width, width, dtype=torch.float32))
            store["hessians"][name] = blocks
        else:
            H_shape = (alloc_f, alloc_f)
            store["hessians"][name] = _allocate_hessian(
                name, H_shape, use_mmap_full, mmap_dir, store
            )

        collector = ActivationStatsCollector(
            layer_name=name,
            store=store,
            layer_type=ltype,
            hessian_block_size=hessian_block_size,
            collect_amax=collect_amax,
            rot_size=rot_size,
            mode=collector_mode,
        )
        collector.register(module)
        collectors.append(collector)

        if use_mmap_full and idx % max(1, len(targets) // 10) == 0:
            logger.info("Allocated %d/%d Hessian buffers (%.1fs)", idx, len(targets), time.time() - alloc_start)

    logger.info("Allocated %d Hessian buffers in %.1fs", len(collectors), time.time() - alloc_start)

    _run_samples(model_patcher, conditioning, num_steps, num_samples, seed,
                 latent_height, latent_width, progress_callback, pass_label="stats")

    for c in collectors:
        c.remove()

    # ── PermuQuant: compute permutations and apply to Hessians ──
    if permuquant:
        mu2_store = store.pop("mu2", {})
        count_store = store.pop("_mu2_count", {})
        for name, mu2_sum in mu2_store.items():
            n = count_store.get(name, 1)
            mu2 = mu2_sum / n
            # mu2 was accumulated after rotation (PermuQuant-H).
            # Compute correct in_f for Conv2d (flattened features).
            shape = shapes[name]
            ltype = layer_types[name]
            if ltype == "Conv2d":
                in_f = shape[1] * shape[2] * shape[3]
            else:
                in_f = shape[1]
            # Crop mu2 to in_f BEFORE sorting so permutation indices stay
            # in [0, in_f-1].  This is required because the converter
            # validates perm.max() < in_f.  Without the crop, indices from
            # the rotated-padded space (up to alloc_f-1) would fail the
            # check and the converter would skip the permutation entirely,
            # leaving the weight unpermuted while the Hessian is permuted.
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

    # Validate Hessians for corruption (extreme outliers, asymmetry)
    _validate_hessians(store.get("hessians", {}))

    # Crop rotation padding: when rot_size > 0, Hessians were allocated at the
    # padded feature count.  Crop back to the original in_features so the saved
    # file matches the actual layer dimensions.
    if rot_size > 0:
        for name, module in _walk_target_modules(target):
            ltype = "Conv2d" if isinstance(module, nn.Conv2d) else "Linear"
            in_f = _get_in_features(module, ltype)
            H = store.get("hessians", {}).get(name)
            if H is None:
                continue
            if isinstance(H, list):
                # block-diagonal list: crop the last block if padded
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

    channels, lat_h, lat_w = _infer_latent_shape(model_patcher, latent_height, latent_width)

    metadata = {
        "model_name": _guess_model_name(model_patcher),
        "num_samples": int(num_samples),
        "num_steps": int(num_steps),
        "seed": int(seed),
        "hessian_block_size": int(hessian_block_size),
        "collect_amax": bool(collect_amax),
        "rot_size": int(rot_size),
        "hessian_rotated": rot_size > 0,
        "permuquant_enabled": permuquant,
        "collection_date": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "recommended_damping_ratio": 0.01,
        "quantization_order": "column",
        "num_layers": len(targets),
        "latent_shape": [int(channels), int(lat_h), int(lat_w)],
    }

    result = {
        "metadata": metadata,
        "hessians": store.get("hessians", {}),
        "amax": store.get("amax", {}) if collect_amax else {},
        "shapes": shapes,
        "layer_types": layer_types,
    }
    if permutations:
        result["permuquant"] = permutations
    if mmap_dir is not None:
        result["_mmap_temp_dir"] = mmap_dir
    return result


def _guess_model_name(model_patcher) -> str:
    inner = getattr(model_patcher, "model", None)
    if inner is None:
        return "unknown"
    cls = inner.__class__.__name__
    config = getattr(inner, "model_config", None)
    if config is not None:
        return f"{cls}:{config.__class__.__name__}"
    return cls
