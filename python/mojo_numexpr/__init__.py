"""A Mojo-backed compatible subset of NumExpr's public Python API."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import numpy as np

from .compiler import (
    ExpressionError,
    Program,
    compile_expression,
    default_threads,
    execute,
    resolve_names,
)

__version__ = "0.1.0"
_num_threads = default_threads()
_last: tuple[Program, dict[str, Any], dict[str, Any]] | None = None


class NumExpr:
    def __init__(self, ex, signature=(), sanitize: bool = True, **kwargs):
        if kwargs:
            unknown = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument {unknown!r}")
        self.expression = ex
        self._program = compile_expression(ex, sanitize)
        self.input_names = self._program.input_names
        self.signature = signature
        self.constants = tuple(self._program.constants)
        self.program = self._program.code.tobytes()

    def __call__(self, *args, **kwargs):
        if kwargs:
            unknown = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument {unknown!r}")
        if len(args) != len(self.input_names):
            raise ValueError(
                f"number of inputs doesn't match program ({len(args)} given, "
                f"expected {len(self.input_names)})"
            )
        return execute(
            self._program,
            dict(zip(self.input_names, args)),
            nthreads=_num_threads,
        )


def evaluate(
    ex: str,
    local_dict: dict | None = None,
    global_dict: dict | None = None,
    out: np.ndarray | None = None,
    order: str = "K",
    casting: str = "same_kind",
    sanitize: bool | None = None,
    _frame_depth: int = 3,
    disable_cache: bool = False,
    **kwargs,
) -> np.ndarray:
    del disable_cache
    if kwargs:
        unknown = next(iter(kwargs))
        raise ValueError(f"Unknown keyword argument {unknown!r}")
    program = compile_expression(ex, True if sanitize is None else sanitize)
    values = resolve_names(program, local_dict, global_dict, 2)
    result = execute(
        program,
        values,
        out=out,
        order=order,
        casting=casting,
        nthreads=_num_threads,
    )
    global _last
    _last = (program, dict(local_dict or {}), dict(global_dict or {}))
    return result


def re_evaluate(
    local_dict: dict | None = None,
    global_dict: dict | None = None,
    _frame_depth: int = 2,
) -> np.ndarray:
    if _last is None:
        raise RuntimeError("no previous evaluate() expression")
    program, previous_local, previous_global = _last
    local_source = local_dict if local_dict is not None else previous_local or None
    global_source = global_dict if global_dict is not None else previous_global or None
    values = resolve_names(program, local_source, global_source, 2)
    return execute(program, values, nthreads=_num_threads)


def get_num_threads() -> int:
    return _num_threads


def set_num_threads(nthreads):
    global _num_threads
    value = int(nthreads)
    if value < 1:
        raise ValueError("nthreads must be a positive integer")
    previous = _num_threads
    _num_threads = min(value, detect_number_of_cores())
    return previous


def detect_number_of_cores() -> int:
    return os.cpu_count() or 1


def detect_number_of_threads() -> int:
    return detect_number_of_cores()


def set_vml_num_threads(nthreads):
    del nthreads
    return None


def set_vml_accuracy_mode(mode):
    if mode not in (None, "low", "high", "fast"):
        raise ValueError("mode must be one of 'low', 'high', 'fast', or None")
    return None


def print_versions() -> None:
    print(f"-= mojo-numexpr version: {__version__} =-")
    print(f"-= NumPy version: {np.__version__} =-")
    print(f"-= Python version: {sys.version.split()[0]} =-")
    print(f"-= Platform: {platform.platform()} =-")
    print(f"-= Number of threads: {_num_threads} =-")


__all__ = [
    "ExpressionError",
    "NumExpr",
    "detect_number_of_cores",
    "detect_number_of_threads",
    "evaluate",
    "get_num_threads",
    "print_versions",
    "re_evaluate",
    "set_num_threads",
    "set_vml_accuracy_mode",
    "set_vml_num_threads",
]
