"""ComfyUI node definitions for calibration data collection.

Exports ``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` for
ComfyUI's node loader.  No weights are modified.
"""
import logging
import os
import shutil
from typing import Dict, Optional, Tuple

import torch

import folder_paths
import comfy.samplers
import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

try:
    from .calibration import collect_stats, collect_stats_dual
    from .utils import estimate_disk_size, human_size, save_calibration, default_dual_output_paths
except ImportError:
    from calibration import collect_stats, collect_stats_dual
    from utils import estimate_disk_size, human_size, save_calibration, default_dual_output_paths


logger = logging.getLogger("comfyui_gptq_calibration")

__all__ = [
    "CalibrationDataCollector",
    "DualModelCalibrationDataCollector",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

# Maximum seed value (32-bit unsigned max; matches ComfyUI's seed convention).
_MAX_SEED: int = 0xFFFFFFFF


# ── Shared helpers ─────────────────────────────────────────────────────────


def _default_output_path() -> str:
    """Return the default calibration output path in ComfyUI's output directory."""
    base = folder_paths.get_output_directory() if hasattr(folder_paths, "get_output_directory") else folder_paths.output_directory
    return os.path.join(base, "calibration.pt")


def _normalize_conditioning(conditioning) -> list:
    """Validate and return ComfyUI CONDITIONING (list of [tensor, dict] entries)."""
    if conditioning is None:
        return []
    if not isinstance(conditioning, list):
        raise ValueError("conditioning must be a list of CONDITIONING entries")
    if len(conditioning) == 0:
        return []
    return conditioning


def _extract_layer_stats(data: Dict) -> Tuple[int, int]:
    """Extract ``(num_layers, avg_in_features)`` from calibration result data."""
    try:
        num_layers = int(data["metadata"]["num_layers"])
    except Exception:
        num_layers = len(data.get("shapes", {}))
    try:
        shapes = data.get("shapes", {})
        def _in_features(s):
            # Conv2d weight: (out_ch, in_ch, kH, kW) → in_ch * kH * kW
            # Linear weight: (out_features, in_features) → in_features
            if len(s) == 4:
                return s[1] * s[2] * s[3]
            return s[1] if len(s) > 1 else s[0]
        avg_in = int(sum(_in_features(s) for s in shapes.values()) / max(1, len(shapes)))
    except Exception:
        avg_in = 0
    return num_layers, avg_in


def _log_disk_estimate(data: Dict, hessian_block_size: int, collect_amax: bool,
                       hessian_format: str, dlr_rank: int, label: str = "") -> None:
    """Log the estimated calibration file size for *data*."""
    num_layers, avg_in = _extract_layer_stats(data)
    estimate = estimate_disk_size(num_layers, avg_in, hessian_block_size, collect_amax,
                                  hessian_format=hessian_format, dlr_rank=dlr_rank)
    logger.info(
        "Estimated calibration file size (%s): %s (Hessian %s, amax %s) for %d layers, avg in_features=%d",
        label,
        human_size(estimate["total_bytes"]),
        human_size(estimate["hessian_bytes"]),
        human_size(estimate["amax_bytes"]),
        num_layers,
        avg_in,
    )


def _cleanup_mmap_dir(data: Dict) -> None:
    """Remove the temporary mmap directory referenced by *data*, if any."""
    mmap_dir = data.pop("_mmap_temp_dir", None)
    if mmap_dir and os.path.isdir(mmap_dir):
        shutil.rmtree(mmap_dir, ignore_errors=True)
        logger.info("Cleaned up temporary Hessian mmap directory %s", mmap_dir)


def _make_progress_callback(label: str):
    """Return a progress callback that logs with the given *label* prefix."""
    def _cb(done, total, msg):
        logger.info("%s: %s", label, msg)
    return _cb


# ── Node definitions ───────────────────────────────────────────────────────


class CalibrationDataCollector(ComfyNodeABC):
    """Collect per-layer Hessians and amax from a loaded diffusion model.

    Output is a ``.pt`` file for external quantizers (GPTQ/OBQ/ConvRot).
    No weights are modified.  The ``collect`` method is the ComfyUI entry
    point (set via ``FUNCTION = "collect"``).

    Side effects: writes a ``.pt`` file to *output_path*; creates and
    cleans up a ``.gptq_hessian_tmp/`` directory for mmap mode.
    """

    DESCRIPTION = (
        "Collect per-layer Hessians (and optionally activation amax) for "
        "external quantization. No weights are modified."
    )
    CATEGORY = "model/quantization"
    FUNCTION = "collect"
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("calibration_path",)
    OUTPUT_NODE = True

    @classmethod
    # ``s`` is the node class per ComfyUI's convention for INPUT_TYPES.
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "Loaded diffusion model (FP16/BF16/FP32)."}),
                "conditioning": (IO.CONDITIONING, {"tooltip": "Pre-encoded conditioning from CLIPTextEncode or similar."}),
                "num_steps": (IO.INT, {"default": 4, "min": 1, "max": 50, "tooltip": "Denoising steps per sample."}),
                "num_samples": (IO.INT, {"default": 16, "min": 1, "max": 4096, "tooltip": "Independent samples to accumulate over. 16-128 recommended."}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": _MAX_SEED, "tooltip": "Seed for noise and timestep sampling."}),
                "hessian_block_size": (IO.INT, {"default": 128, "min": 0, "max": 1024, "tooltip": "0 = full H (paper-accurate, auto memory-mapped to disk). 128 = diagonal blocks (default, saves RAM). Ignored when hessian_format='dlr'."}),
                "hessian_format": ("COMBO", {"options": ["block", "full", "dlr"], "default": "dlr", "tooltip": "Hessian storage format. 'block' = diagonal blocks (use hessian_block_size). 'full' = full Hessian (memory-mapped). 'dlr' = Diagonal + Low-Rank via Frequent Directions (same memory as block, captures cross-block correlations). Recommended default."}),
                "dlr_rank": (IO.INT, {"default": 128, "min": 1, "max": 4096, "tooltip": "Rank for DLR Hessian (only used when hessian_format='dlr'). Same memory budget as block_size=rank. Recommended: 64-256."}),
                "collect_amax": (IO.BOOLEAN, {"default": True, "tooltip": "Also collect max(abs(x)) per layer. Required for activation quantization; not used by weight-only GPTQ."}),
                "output_path": (IO.STRING, {"default": _default_output_path(), "tooltip": "Where to save the calibration .pt file."}),
            },
            "optional": {
                "latent_height": (IO.INT, {"default": 64, "min": 8, "max": 1024, "tooltip": "Latent spatial height. 128 for 1024px, 64 for 512px."}),
                "latent_width": (IO.INT, {"default": 64, "min": 8, "max": 1024, "tooltip": "Latent spatial width. 128 for 1024px, 64 for 512px."}),
                "convrot": (IO.BOOLEAN, {"default": False, "tooltip": "Enable ConvRot Hadamard rotation. Collects Hessians in rotated space for better block-diagonal approximation."}),
                "rot_size": (IO.INT, {"default": 256, "min": 16, "max": 4096, "tooltip": "Hadamard group size (must be power of 2). 256 recommended for ConvRot."}),
                "permuquant": (IO.BOOLEAN, {"default": False, "tooltip": "Enable PermuQuant channel reordering. Runs a second calibration pass with channels sorted by second moment for better quantization."}),
                "piso": (IO.BOOLEAN, {"default": False, "tooltip": "Collect Hessian diagonal for PiSO data-aware scale optimization. Adds a small overhead to store diag(X^T X) per layer, which the converter uses to compute optimal per-row scales instead of absmax."}),
                "sigma_min": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "Lower bound of the sigma range to sample. Set to 0.875 with Wan 2.2 high-noise expert, or 0.0 for full range (default)."}),
                "sigma_max": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "Upper bound of the sigma range to sample. Set to 0.875 with Wan 2.2 low-noise expert, or 1.0 for full range (default)."}),
                "force_cpu_hook": (IO.BOOLEAN, {"default": False, "tooltip": "Force hook-side processing to CPU. Enable only if you hit GPU OOM during calibration — the GPU-fast path uses ~20-50 MB of transient VRAM per layer."}),
            },
        }

    def collect(self,
                model,
                conditioning,
                num_steps: int,
                num_samples: int,
                seed: int,
                hessian_block_size: int,
                hessian_format: str,
                dlr_rank: int,
                collect_amax: bool,
                output_path: str,
                latent_height: int = 64,
                latent_width: int = 64,
                convrot: bool = False,
                rot_size: int = 256,
                permuquant: bool = False,
                piso: bool = False,
                sigma_min: float = 0.0,
                sigma_max: float = 1.0,
                force_cpu_hook: bool = False) -> Tuple[str]:
        """Collect calibration data and save to ``output_path``.

        Returns ``(calibration_path,)`` for ComfyUI output routing.
        """
        cond = _normalize_conditioning(conditioning)
        if not cond:
            raise ValueError("conditioning is empty")

        progress = comfy.utils.ProgressBar(num_samples)
        def _cb(done, total, msg):
            progress.update_absolute(done, total)
            logger.info("Calibration: %s", msg)

        data = collect_stats(
            model_patcher=model,
            conditioning=cond,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed,
            latent_height=latent_height,
            latent_width=latent_width,
            hessian_block_size=hessian_block_size,
            collect_amax=collect_amax,
            rot_size=rot_size if convrot else 0,
            output_path=output_path,
            progress_callback=_cb,
            permuquant=permuquant,
            piso=piso,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            hessian_format=hessian_format,
            dlr_rank=dlr_rank,
            force_cpu_hook=force_cpu_hook,
        )

        try:
            _log_disk_estimate(data, hessian_block_size, collect_amax,
                               hessian_format, dlr_rank, label=output_path)

            path = save_calibration(data, output_path)
            logger.info("Calibration data written to %s", path)
        finally:
            _cleanup_mmap_dir(data)

        progress.update_absolute(num_samples, num_samples)
        return (path,)


def _normalize_dual_negative(negative) -> list:
    """Normalise the negative conditioning for the dual-model node.

    ``None`` is the Ideogram 4 image-only default — the model receives no
    text context and runs in ``_run_image_only`` mode.  This is represented
    as ``[[None, {}]]`` in ComfyUI's conditioning format.  Any other value
    is validated like normal conditioning.
    """
    if negative is None:
        return [[None, {}]]
    return _normalize_conditioning(negative)


def _default_dual_positive_path() -> str:
    pos, _ = default_dual_output_paths()
    return pos


def _default_dual_negative_path() -> str:
    _, neg = default_dual_output_paths()
    return neg


class DualModelCalibrationDataCollector(ComfyNodeABC):
    """Calibrate two models used together via dual-model CFG (e.g. Ideogram 4).

    Mirrors ``DualModelGuider``: positive conditioning runs through *model*,
    negative conditioning (often image-only) runs through *model_negative*,
    and CFG is applied between them at each step.

    Outputs two ``.pt`` files — one per model — each in the same schema as
    the single-model ``CalibrationDataCollector`` output.

    Side effects: writes two ``.pt`` files; creates and cleans up
    ``.gptq_hessian_tmp/`` directories for mmap mode.  No weights modified.
    """

    DESCRIPTION = (
        "Calibrate two models used together via dual-model CFG.  "
        "Outputs two .pt files (one per model) for external quantization."
    )
    CATEGORY = "model/quantization"
    FUNCTION = "collect"
    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("calibration_path_positive", "calibration_path_negative")
    OUTPUT_NODE = True

    @classmethod
    # ``s`` is the node class per ComfyUI's convention for INPUT_TYPES.
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "Positive (conditional) model."}),
                "model_negative": (IO.MODEL, {"tooltip": "Negative (unconditional) model. For Ideogram 4 this is the image-only expert."}),
                "positive": (IO.CONDITIONING, {"tooltip": "Positive conditioning — runs through `model`."}),
                "negative": (IO.CONDITIONING, {"optional": True, "tooltip": "Negative conditioning for the uncond model. Leave disconnected for image-only pass (default for Ideogram 4)."}),
                "cfg": (IO.FLOAT, {"default": 4.0, "min": 1.01, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "CFG value to apply between the two models. Must be > 1.0 to calibrate both models."}),
                "num_steps": (IO.INT, {"default": 4, "min": 1, "max": 50}),
                "num_samples": (IO.INT, {"default": 16, "min": 1, "max": 4096}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": _MAX_SEED}),
                "hessian_block_size": (IO.INT, {"default": 128, "min": 0, "max": 1024, "tooltip": "0 = full Hessian, 128 = diagonal blocks. Ignored when hessian_format='dlr'."}),
                "hessian_format": ("COMBO", {"options": ["block", "full", "dlr"], "default": "dlr", "tooltip": "Hessian storage format. 'dlr' = Diagonal + Low-Rank (same memory as block, captures cross-block correlations). Recommended default."}),
                "dlr_rank": (IO.INT, {"default": 128, "min": 1, "max": 4096, "tooltip": "Rank for DLR Hessian (only used when hessian_format='dlr')."}),
                "collect_amax": (IO.BOOLEAN, {"default": True}),
                "output_path_positive": (IO.STRING, {"default": _default_dual_positive_path(), "tooltip": "Where to save the positive model's calibration .pt file."}),
                "output_path_negative": (IO.STRING, {"default": _default_dual_negative_path(), "tooltip": "Where to save the negative model's calibration .pt file."}),
            },
            "optional": {
                "latent_height": (IO.INT, {"default": 64, "min": 8, "max": 1024}),
                "latent_width": (IO.INT, {"default": 64, "min": 8, "max": 1024}),
                "convrot": (IO.BOOLEAN, {"default": False}),
                "rot_size": (IO.INT, {"default": 256, "min": 16, "max": 4096}),
                "permuquant": (IO.BOOLEAN, {"default": False}),
                "piso": (IO.BOOLEAN, {"default": False}),
                "sigma_min": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "Lower bound of the sigma range to sample. Set to 0.875 with Wan 2.2 high-noise expert, or 0.0 for full range (default)."}),
                "sigma_max": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "tooltip": "Upper bound of the sigma range to sample. Set to 0.875 with Wan 2.2 low-noise expert, or 1.0 for full range (default)."}),
                "force_cpu_hook": (IO.BOOLEAN, {"default": False, "tooltip": "Force hook-side processing to CPU. Enable if you hit GPU OOM with dual-model calibration."}),
            },
        }

    def collect(self,
                model,
                model_negative,
                positive,
                cfg: float,
                num_steps: int,
                num_samples: int,
                seed: int,
                hessian_block_size: int,
                hessian_format: str,
                dlr_rank: int,
                collect_amax: bool,
                output_path_positive: str,
                output_path_negative: str,
                negative=None,
                latent_height: int = 64,
                latent_width: int = 64,
                convrot: bool = False,
                rot_size: int = 256,
                permuquant: bool = False,
                piso: bool = False,
                sigma_min: float = 0.0,
                sigma_max: float = 1.0,
                force_cpu_hook: bool = False) -> Tuple[str, str]:
        """Collect calibration data for both models and save to their paths.

        Returns ``(calibration_path_positive, calibration_path_negative)``
        for ComfyUI output routing.
        """
        # Resolve negative: None → [[None, {}]] (image-only)
        neg = _normalize_dual_negative(negative)
        pos = _normalize_conditioning(positive)
        if not pos:
            raise ValueError("positive conditioning is empty")

        # Validate paths are different
        if os.path.realpath(output_path_positive) == os.path.realpath(output_path_negative):
            raise ValueError(
                "output_path_positive and output_path_negative must be different"
            )

        progress = comfy.utils.ProgressBar(num_samples)
        def _cb(done, total, msg):
            progress.update_absolute(done, total)
            logger.info("Dual calibration: %s", msg)

        data_pos, data_neg = collect_stats_dual(
            model_patcher=model,
            model_negative_patcher=model_negative,
            positive=pos,
            negative=neg,
            cfg=cfg,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed,
            latent_height=latent_height,
            latent_width=latent_width,
            hessian_block_size=hessian_block_size,
            collect_amax=collect_amax,
            rot_size=rot_size if convrot else 0,
            output_path_positive=output_path_positive,
            output_path_negative=output_path_negative,
            progress_callback=_cb,
            permuquant=permuquant,
            piso=piso,
            hessian_format=hessian_format,
            dlr_rank=dlr_rank,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            force_cpu_hook=force_cpu_hook,
        )

        try:
            path_pos = save_calibration(data_pos, output_path_positive)
            path_neg = save_calibration(data_neg, output_path_negative)
            logger.info("Dual calibration written to %s and %s", path_pos, path_neg)

            _log_disk_estimate(data_pos, hessian_block_size, collect_amax,
                               hessian_format, dlr_rank, label=path_pos)
            _log_disk_estimate(data_neg, hessian_block_size, collect_amax,
                               hessian_format, dlr_rank, label=path_neg)
        finally:
            for data in (data_pos, data_neg):
                _cleanup_mmap_dir(data)

        progress.update_absolute(num_samples, num_samples)
        return (path_pos, path_neg)


NODE_CLASS_MAPPINGS = {
    "CalibrationDataCollector": CalibrationDataCollector,
    "DualModelCalibrationDataCollector": DualModelCalibrationDataCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CalibrationDataCollector": "Calibration Data Collector",
    "DualModelCalibrationDataCollector": "Dual Model Calibration Data Collector",
}
