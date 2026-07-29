from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_SURPRISE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-surprise.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msu_svd_train": ([I] * 10 + [F, I] + [F] * 8, None),
    "msu_svdpp_train": ([I] * 14 + [F] * 11, None),
    "msu_similarities": ([I] * 9 + [F, F, I], None),
    "msu_baseline_als": ([I] * 11 + [F, F, F], None),
    "msu_baseline_sgd": ([I] * 7 + [F, F, F], None),
    "msu_knn_predict": ([I] * 18 + [F], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_SURPRISE_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_SURPRISE_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_SURPRISE_LIB") and os.path.exists(LIB) and not force:
        return LIB
    source = os.path.join(SRC, "kernels.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = _mojo_command() + ["build", "--emit", "shared-lib", source, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None
_cpu_device: int | None = None


def _release_cpu_device() -> None:
    global _cpu_device
    if _loaded is not None and _cpu_device is not None:
        release = _loaded.KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice
        release.argtypes = [ctypes.c_void_p]
        release(_cpu_device)
        _cpu_device = None


def parallel_available() -> bool:
    global _cpu_device
    native = lib()
    if _cpu_device is None:
        try:
            initialize = native.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
            initialize.argtypes = []
            initialize.restype = ctypes.c_void_p
            _cpu_device = initialize()
        except (AttributeError, OSError):
            return False
    return _cpu_device is not None


atexit.register(_release_cpu_device)


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def addr(
    array: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
    min_size: int = 0,
    writable: bool = False,
) -> int:
    """Return an ABI-safe address while the caller keeps ``array`` alive."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise TypeError(f"FFI buffer must have dtype {np.dtype(dtype)}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffer must be C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output buffer must be writable")
    if array.size < min_size:
        raise ValueError(
            f"FFI buffer has {array.size} elements, expected at least {min_size}"
        )
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffer has a null address")
    return address


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)
