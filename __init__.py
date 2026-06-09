"""ComfyUI custom node plugin for collecting per-layer activation
statistics (Hessians and optional amax) for external quantization tools.

The plugin is intentionally **read-only with respect to model weights**:
no quantization, INT4 packing or safetensors output is produced. The
single ``.pt`` artifact emitted is meant to be consumed offline by a
GPTQ/OBQ/ConvRot-style tool.
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
