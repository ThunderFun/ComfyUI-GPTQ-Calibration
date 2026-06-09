"""Smoke tests for the node definitions and public mappings.

These tests verify that the ``NODE_CLASS_MAPPINGS`` and
``NODE_DISPLAY_NAME_MAPPINGS`` dicts have the expected keys.
"""
from __future__ import annotations

from unittest import mock


def test_node_class_mappings_keys_present():
    expected_nodes = {
        "CalibrationDataCollector",
    }

    with mock.patch.dict("sys.modules", {
        "folder_paths": mock.MagicMock(get_output_directory=lambda: "/tmp"),
        "comfy": mock.MagicMock(),
        "comfy.samplers": mock.MagicMock(),
        "comfy.utils": mock.MagicMock(),
        "comfy.comfy_types": mock.MagicMock(),
        "comfy.comfy_types.IO": mock.MagicMock(),
    }):
        # Import the package which re-exports the mappings.
        from comfyui_gptq_calibration import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

    assert expected_nodes.issubset(NODE_CLASS_MAPPINGS.keys())
    for name in expected_nodes:
        assert name in NODE_DISPLAY_NAME_MAPPINGS
        assert NODE_DISPLAY_NAME_MAPPINGS[name]  # non-empty display name
