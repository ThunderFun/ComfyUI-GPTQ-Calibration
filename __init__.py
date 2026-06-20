"""Collect per-layer activation statistics (Hessians, amax) for external
quantization tools (GPTQ/OBQ/ConvRot).  Read-only: no weights modified.
"""

__version__ = "0.1.0"

# Dual import-path fallback: ComfyUI loads custom_nodes via two mechanisms.
# 1. As a package (``from comfyui_gptq_calibration import ...``) — uses relative import.
# 2. As a flat module in ``custom_nodes/`` — uses absolute import.
# Both must resolve NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
