"""ctypes access to the prebuilt Mojo expression evaluator."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_NUMEXPR_LIB") or os.path.join(
    ROOT, "dist", "libmojo-numexpr.so"
)

I = ctypes.c_int64
_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build() -> str:
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            build()
        _library = ctypes.CDLL(LIB)
        _library.mne_evaluate.argtypes = [I] * 12
        _library.mne_evaluate.restype = I
        _library.mne_mul_add_f64.argtypes = [I] * 5
        _library.mne_mul_add_f64.restype = None
        _library.mne_polynomial_f64.argtypes = [I] * 5
        _library.mne_polynomial_f64.restype = None
        _library.mne_transcendental_f64.argtypes = [I] * 5
        _library.mne_transcendental_f64.restype = None
        _library.mne_conditional_f64.argtypes = [I] * 5
        _library.mne_conditional_f64.restype = None
    return _library


def addr(value: np.ndarray) -> int:
    address = value.ctypes.data
    if not address:
        raise ValueError("cannot pass a null NumPy buffer across the Mojo FFI")
    return address
