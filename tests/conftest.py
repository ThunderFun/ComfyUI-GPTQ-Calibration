"""Shared pytest fixtures and module-level stubs.

ComfyUI's runtime modules (``comfy.*``, ``folder_paths``) are not
installable on their own, so the test suite uses stubs to allow
importing the calibration modules. The stubs are installed at *import
time* (not in a fixture) so they are present before pytest begins
collecting test modules.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT


# ---------------------------------------------------------------------------
# Install comfy.* / folder_paths stubs at import time so that importing
# the calibration modules does not fail.
# ---------------------------------------------------------------------------

def _install_stub(name: str, package: bool = False) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    if package:
        mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


# ``comfy`` must be a *package* (have ``__path__``) so that submodule
# imports like ``import comfy.sample`` succeed and so that
# attribute access on the parent (``comfy.sample``) resolves. We also
# pre-bind each submodule as an attribute of ``comfy`` to be safe.
_comfy = _install_stub("comfy", package=True)
for _name in (
    "comfy.samplers",
    "comfy.sample",
    "comfy.model_management",
    "comfy.utils",
    "comfy.comfy_types",
):
    leaf = _name.rsplit(".", 1)[-1]
    sub = _install_stub(_name)
    setattr(_comfy, leaf, sub)
_install_stub("folder_paths")

# Build a tiny ``IO`` / ``InputTypeDict`` / ``ComfyNodeABC`` shim so the
# node module can import ``from comfy.comfy_types import IO, ComfyNodeABC,
# InputTypeDict``. We don't exercise the runtime behaviour in these tests;
# we only assert the public name mappings are present.
class _IO(str):
    pass

for _sym in ("MODEL", "CLIP", "CONDITIONING", "STRING", "INT", "BOOLEAN", "FLOAT"):
    setattr(_IO, _sym, _sym)

class _ComfyNodeABC:
    pass

_InputTypeDict = dict

sys.modules["comfy.comfy_types"].IO = _IO
sys.modules["comfy.comfy_types"].ComfyNodeABC = _ComfyNodeABC
sys.modules["comfy.comfy_types"].InputTypeDict = _InputTypeDict

sys.modules["comfy.samplers"].simple_scheduler = (
    lambda model_sampling, steps: torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
)
sys.modules["comfy.samplers"].sample = lambda *args, **kwargs: None
sys.modules["comfy.sample"].sample = lambda *args, **kwargs: None
sys.modules["comfy.model_management"].soft_empty_cache = lambda: None
sys.modules["comfy.utils"].ProgressBar = lambda total, node_id=None: types.SimpleNamespace(
    update_absolute=lambda *a, **k: None
)
sys.modules["folder_paths"].get_output_directory = lambda: str(REPO_ROOT / "tests" / "_output")
sys.modules["folder_paths"].output_directory = str(REPO_ROOT / "tests" / "_output")


# ---------------------------------------------------------------------------
# Pre-load the real modules so tests can import them via the
# comfyui_gptq_calibration namespace.
# ---------------------------------------------------------------------------

_pkg_stub = _install_stub("comfyui_gptq_calibration")
_pkg_stub.__path__ = [str(PKG_DIR)]  # type: ignore[attr-defined]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_utils_mod = _load("comfyui_gptq_calibration._utils", PKG_DIR / "utils.py")
_calibration_mod = _load("comfyui_gptq_calibration._calibration", PKG_DIR / "calibration.py")
_nodes_mod = _load("comfyui_gptq_calibration._nodes", PKG_DIR / "nodes.py")
# Re-export under the public name so ``from comfyui_gptq_calibration.utils
# import ...`` works in test modules.
sys.modules["comfyui_gptq_calibration.utils"] = _utils_mod
sys.modules["comfyui_gptq_calibration.calibration"] = _calibration_mod
sys.modules["comfyui_gptq_calibration.nodes"] = _nodes_mod

# Make ``from comfyui_gptq_calibration import NODE_CLASS_MAPPINGS, ...``
# work without running the package's real __init__ (which would re-import
# the real ``comfy.comfy_types`` and fail in this stubbed environment).
_pkg_stub.NODE_CLASS_MAPPINGS = _nodes_mod.NODE_CLASS_MAPPINGS
_pkg_stub.NODE_DISPLAY_NAME_MAPPINGS = _nodes_mod.NODE_DISPLAY_NAME_MAPPINGS


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def utils_mod():
    return _utils_mod


@pytest.fixture(scope="session")
def calibration_mod():
    return _calibration_mod
