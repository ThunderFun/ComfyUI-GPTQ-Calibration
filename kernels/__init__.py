"""Triton FHT rotation kernels for calibration.

Provides O(N log N) Fast Hadamard Transform rotation as a drop-in
replacement for the O(N²) matmul-based rotate_activations().
"""

from .triton_fht_rotate import fht_rotate, _HAS_TRITON

__all__ = ["fht_rotate", "_HAS_TRITON"]
