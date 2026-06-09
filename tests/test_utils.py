"""Tests for calibration utilities."""
from __future__ import annotations

import torch

from comfyui_gptq_calibration.utils import (
    estimate_disk_size,
    human_size,
    load_calibration,
    save_calibration,
)


def test_estimate_disk_size_full_hessian():
    e = estimate_disk_size(num_layers=100, avg_in_features=3072, hessian_block_size=0, collect_amax=False)
    expected = 100 * 3072 * 3072 * 4
    assert e["hessian_bytes"] == expected
    assert e["amax_bytes"] == 0
    assert e["total_bytes"] == expected
    assert e["per_layer_bytes"] == 3072 * 3072 * 4


def test_estimate_disk_size_block_hessian():
    e = estimate_disk_size(num_layers=100, avg_in_features=3072, hessian_block_size=128, collect_amax=True)
    num_blocks = (3072 + 127) // 128  # 24
    expected_h = 100 * num_blocks * 128 * 128 * 4
    assert e["hessian_bytes"] == expected_h
    assert e["amax_bytes"] == 100 * 4
    assert e["total_bytes"] == expected_h + 100 * 4
    assert e["per_layer_bytes"] == num_blocks * 128 * 128 * 4


def test_human_size_formats():
    assert human_size(0).endswith("B")
    assert human_size(1024).startswith("1.00 KB")
    assert human_size(1024 ** 2).startswith("1.00 MB")
    assert human_size(1024 ** 3).startswith("1.00 GB")


def test_save_and_load_roundtrip(tmp_path):
    payload = {
        "metadata": {"a": 1, "b": [1, 2, 3]},
        "hessians": {"layer": torch.zeros(4, 4)},
        "amax": {"layer": 0.5},
    }
    p = tmp_path / "calib.pt"
    returned = save_calibration(payload, str(p))
    assert returned == str(p)
    assert p.is_file()

    loaded = load_calibration(str(p))
    assert loaded["metadata"]["a"] == 1
    assert torch.equal(loaded["hessians"]["layer"], torch.zeros(4, 4))
    assert loaded["amax"]["layer"] == 0.5


def test_save_calibration_creates_parent_directory(tmp_path):
    target = tmp_path / "nested" / "subdir" / "out.pt"
    save_calibration({"x": 1}, str(target))
    assert target.is_file()
