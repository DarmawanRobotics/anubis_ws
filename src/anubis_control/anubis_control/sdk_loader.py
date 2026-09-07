"""SDK loading for anubis_control.

Single, explicit strategy: the SDK lives at $ (set in the
workspace Dockerfile -- see docker/Dockerfile). No directory-climbing
fallback chain like darmawan_ws had; if the env var is missing or wrong,
this fails loudly and immediately instead of silently guessing.
"""

import importlib
import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional


DEFAULT_SDK_DIR = "/opt/genisom_l1_sdk"


class SdkLoadError(RuntimeError):
    """Raised when the SDK module cannot be located or imported."""


@dataclass
class SdkHandle:
    module: object
    model: str
    arch: str
    lib_path: str


def _normalized_arch() -> str:
    machine = platform.machine()
    return machine.replace("amd64", "x86_64").replace("arm64", "aarch64")


def load_sdk(model: str = "zsl-1w", sdk_dir: Optional[str] = None) -> SdkHandle:
    """Import the compiled SDK module for `model` ("zsl-1" or "zsl-1w").

    Raises SdkLoadError with a specific, actionable message on failure
    rather than returning None -- callers should let this propagate
    during node startup; a robot control node with no SDK is not in a
    safe state to keep running.
    """
    sdk_dir = sdk_dir or os.environ.get("ANUBIS_SDK_DIR", DEFAULT_SDK_DIR)
    arch = _normalized_arch()
    lib_path = os.path.join(sdk_dir, "lib", model, arch)

    if not os.path.isdir(sdk_dir):
        raise SdkLoadError(
            f"SDK directory not found: {sdk_dir}\n"
            f"  Set ANUBIS_SDK_DIR, or clone the SDK there:\n"
            f"  git clone https://github.com/zsibot/genisom_l1_sdk_old.git {sdk_dir}"
        )
    if not os.path.isdir(lib_path):
        raise SdkLoadError(
            f"SDK model '{model}' not found for architecture '{arch}' at: {lib_path}\n"
            f"  Available models under {sdk_dir}/lib: "
            f"{os.listdir(os.path.join(sdk_dir, 'lib')) if os.path.isdir(os.path.join(sdk_dir, 'lib')) else '(none)'}"
        )

    module_name = f"mc_sdk_{model.replace('-', '_')}_py"
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{ld_path}" if ld_path else lib_path

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SdkLoadError(
            f"Found {lib_path} but could not import {module_name}: {exc}\n"
            f"  This SDK's compiled Python binding requires Python 3.10 "
            f"(running: {sys.version_info.major}.{sys.version_info.minor})"
        ) from exc

    return SdkHandle(module=module, model=model, arch=arch, lib_path=lib_path)
