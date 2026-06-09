import os
import torch


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
    """Load calibration data from a .pt file. Always mapped to CPU.

    Uses ``weights_only=True`` which is sufficient for our payload
    (tensors, dicts, lists, and primitive scalars produced by
    ``collect_stats``) and avoids the PyTorch 2.6+ security warning.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def estimate_disk_size(num_layers: int,
                       avg_in_features: int,
                       hessian_block_size: int = 0,
                       collect_amax: bool = True,
                       dtype_bytes: int = 4) -> dict:
    """Estimate the output file size in bytes.

    Args:
        num_layers: Number of layers that will be instrumented.
        avg_in_features: Average ``in_features`` (or ``C_in*kH*kW`` for conv2d).
        hessian_block_size: ``0`` for full Hessian, else diagonal blocks.
        collect_amax: Whether amax scalars are stored.
        dtype_bytes: Bytes per float (Hessians are accumulated in float32).

    Returns:
        Dict with ``hessian_bytes``, ``amax_bytes``, ``total_bytes`` and
        ``total_gb`` keys.
    """
    if hessian_block_size and hessian_block_size > 0:
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
