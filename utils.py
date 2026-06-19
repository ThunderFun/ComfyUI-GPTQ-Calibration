import logging
import os

import torch

logger = logging.getLogger("comfyui_gptq_calibration")


def default_dual_output_paths(base_path: str | None = None) -> tuple[str, str]:
    """Derive ``<base>_positive.pt`` and ``<base>_negative.pt`` output paths.

    If *base_path* is ``None``, uses the ComfyUI output directory.
    If *base_path* ends with ``.pt``, the suffix is stripped and replaced.
    Raises ``ValueError`` if the two paths resolve to the same location.
    """
    try:
        import folder_paths
        default_dir = folder_paths.get_output_directory() if hasattr(folder_paths, "get_output_directory") else folder_paths.output_directory
    except ImportError:
        default_dir = "/tmp"

    if base_path is None or not base_path.strip():
        base = os.path.join(default_dir, "calibration")
    else:
        base = base_path
        if base.endswith(".pt"):
            base = base[:-3]

    pos = f"{base}_positive.pt"
    neg = f"{base}_negative.pt"

    if os.path.realpath(pos) == os.path.realpath(neg):
        raise ValueError(
            f"output_path_positive and output_path_negative resolve to the same "
            f"path: {os.path.realpath(pos)}.  Use different paths."
        )
    return pos, neg


def save_calibration(data: dict, path: str) -> str:
    """Save calibration data to a .pt file.

    The directory is created if it does not exist. The path is returned
    so callers can chain a (path,) tuple back to ComfyUI.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    torch.save(data, path)
    return path


def load_calibration(path: str) -> dict:
    """Load calibration data from a .pt file (CPU, weights_only=True)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def estimate_disk_size(num_layers: int,
                       avg_in_features: int,
                       hessian_block_size: int = 0,
                       collect_amax: bool = True,
                       dtype_bytes: int = 4,
                       hessian_format: str = "block",
                       dlr_rank: int = 0) -> dict:
    """Estimate output file size. Returns dict with hessian_bytes, amax_bytes, total_bytes, total_gb."""
    if hessian_format == "dlr" and dlr_rank > 0:
        rank = min(dlr_rank, avg_in_features)
        # DLR stores D (n floats) + U (n * rank floats)
        per_layer = (avg_in_features + avg_in_features * rank) * dtype_bytes
    elif hessian_block_size and hessian_block_size > 0:
        block = hessian_block_size
        num_blocks = max(1, (avg_in_features + block - 1) // block)
        per_layer = num_blocks * block * block * dtype_bytes
    else:
        per_layer = avg_in_features * avg_in_features * dtype_bytes

    hessian_bytes = num_layers * per_layer
    amax_bytes = num_layers * dtype_bytes if collect_amax else 0
    total_bytes = hessian_bytes + amax_bytes
    return {
        "hessian_bytes": hessian_bytes,
        "amax_bytes": amax_bytes,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / (1024 ** 3),
        "per_layer_bytes": per_layer,
    }


def human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"
