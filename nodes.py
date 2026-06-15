import logging
import os
import shutil

import torch

import folder_paths
import comfy.samplers
import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

try:
    from .calibration import collect_stats
    from .utils import estimate_disk_size, human_size, save_calibration
except ImportError:
    from calibration import collect_stats
    from utils import estimate_disk_size, human_size, save_calibration


logger = logging.getLogger("comfyui_gptq_calibration")


def _default_output_path() -> str:
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



class CalibrationDataCollector(ComfyNodeABC):
    """Collect per-layer Hessians and amax from a loaded diffusion model.
    Output is a ``.pt`` file for external quantizers (GPTQ/OBQ/ConvRot).
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
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "Loaded diffusion model (FP16/BF16/FP32)."}),
                "conditioning": (IO.CONDITIONING, {"tooltip": "Pre-encoded conditioning from CLIPTextEncode or similar."}),
                "num_steps": (IO.INT, {"default": 4, "min": 1, "max": 50, "tooltip": "Denoising steps per sample."}),
                "num_samples": (IO.INT, {"default": 16, "min": 1, "max": 4096, "tooltip": "Independent samples to accumulate over. 16-128 recommended."}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Seed for noise and timestep sampling."}),
                "hessian_block_size": (IO.INT, {"default": 128, "min": 0, "max": 1024, "tooltip": "0 = full H (paper-accurate, auto memory-mapped to disk). 128 = diagonal blocks (default, saves RAM)."}),
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
            },
        }

    def collect(self,
                model,
                conditioning,
                num_steps: int,
                num_samples: int,
                seed: int,
                hessian_block_size: int,
                collect_amax: bool,
                output_path: str,
                latent_height: int = 64,
                latent_width: int = 64,
                convrot: bool = False,
                rot_size: int = 256,
                permuquant: bool = False,
                piso: bool = False) -> tuple:
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
        )

        try:
            num_layers = int(data["metadata"]["num_layers"])
        except Exception:
            num_layers = len(data.get("shapes", {}))
        try:
            shapes = data.get("shapes", {})
            avg_in = int(sum(s[1] if len(s) > 1 else s[0] for s in shapes.values()) / max(1, len(shapes)))
        except Exception:
            avg_in = 0
        estimate = estimate_disk_size(num_layers, avg_in, hessian_block_size, collect_amax)
        logger.info(
            "Estimated calibration file size: %s (Hessian %s, amax %s) for %d layers, avg in_features=%d",
            human_size(estimate["total_bytes"]),
            human_size(estimate["hessian_bytes"]),
            human_size(estimate["amax_bytes"]),
            num_layers,
            avg_in,
        )

        path = save_calibration(data, output_path)
        logger.info("Calibration data written to %s", path)

        mmap_dir = data.pop("_mmap_temp_dir", None)
        if mmap_dir and os.path.isdir(mmap_dir):
            shutil.rmtree(mmap_dir, ignore_errors=True)
            logger.info("Cleaned up temporary Hessian mmap directory %s", mmap_dir)

        progress.update_absolute(num_samples, num_samples)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "CalibrationDataCollector": CalibrationDataCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CalibrationDataCollector": "Calibration Data Collector",
}
