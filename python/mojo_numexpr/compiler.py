"""Safe AST lowering and execution for the supported NumExpr expression subset."""

from __future__ import annotations

import ast
import inspect
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ._lib import addr, lib

LOAD_VAR, LOAD_CONST = 1, 2
ADD, SUB, MUL, DIV, POW, MOD = range(10, 16)
NEG, ABS, NOT = range(20, 23)
LT, LE, GT, GE, EQ, NE, AND, OR = range(30, 38)
(
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    ATAN2,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    EXP,
    EXPM1,
    LOG,
    LOG1P,
    LOG10,
    SQRT,
    FLOOR,
    CEIL,
    MINIMUM,
    MAXIMUM,
    WHERE,
) = range(40, 64)

_BINARY = {
    ast.Add: ADD,
    ast.Sub: SUB,
    ast.Mult: MUL,
    ast.Div: DIV,
    ast.Pow: POW,
    ast.Mod: MOD,
    ast.BitAnd: AND,
    ast.BitOr: OR,
}
_COMPARE = {
    ast.Lt: LT,
    ast.LtE: LE,
    ast.Gt: GT,
    ast.GtE: GE,
    ast.Eq: EQ,
    ast.NotEq: NE,
}
_FUNCTIONS = {
    "sin": (SIN, 1),
    "cos": (COS, 1),
    "tan": (TAN, 1),
    "arcsin": (ASIN, 1),
    "asin": (ASIN, 1),
    "arccos": (ACOS, 1),
    "acos": (ACOS, 1),
    "arctan": (ATAN, 1),
    "atan": (ATAN, 1),
    "arctan2": (ATAN2, 2),
    "atan2": (ATAN2, 2),
    "sinh": (SINH, 1),
    "cosh": (COSH, 1),
    "tanh": (TANH, 1),
    "arcsinh": (ASINH, 1),
    "asinh": (ASINH, 1),
    "arccosh": (ACOSH, 1),
    "acosh": (ACOSH, 1),
    "arctanh": (ATANH, 1),
    "atanh": (ATANH, 1),
    "exp": (EXP, 1),
    "expm1": (EXPM1, 1),
    "log": (LOG, 1),
    "log1p": (LOG1P, 1),
    "log10": (LOG10, 1),
    "sqrt": (SQRT, 1),
    "floor": (FLOOR, 1),
    "ceil": (CEIL, 1),
    "abs": (ABS, 1),
    "minimum": (MINIMUM, 2),
    "maximum": (MAXIMUM, 2),
    "where": (WHERE, 3),
}
_SUPPORTED_DTYPES = {
    np.dtype(np.float64): 0,
    np.dtype(np.float32): 1,
    np.dtype(np.int64): 2,
    np.dtype(np.int32): 3,
    np.dtype(np.uint8): 4,
    np.dtype(np.bool_): 4,
}
_PARALLEL_ELEMENTS = 1_048_576
_pool: ThreadPoolExecutor | None = None
_SPECIALIZED_CODES = {
    (
        (LOAD_VAR, 0),
        (LOAD_VAR, 1),
        (MUL, 0),
        (LOAD_VAR, 2),
        (ADD, 0),
    ): "mne_mul_add_f64",
    (
        (LOAD_VAR, 0),
        (LOAD_VAR, 1),
        (LOAD_VAR, 2),
        (MUL, 0),
        (ADD, 0),
        (LOAD_VAR, 0),
        (LOAD_VAR, 0),
        (MUL, 0),
        (ADD, 0),
        (LOAD_VAR, 1),
        (LOAD_VAR, 1),
        (MUL, 0),
        (SUB, 0),
        (LOAD_VAR, 2),
        (LOAD_VAR, 2),
        (MUL, 0),
        (LOAD_CONST, 0),
        (MUL, 0),
        (ADD, 0),
    ): "mne_polynomial_f64",
    (
        (LOAD_VAR, 0),
        (SIN, 0),
        (LOAD_VAR, 1),
        (COS, 0),
        (ADD, 0),
        (LOAD_VAR, 2),
        (ABS, 0),
        (NEG, 0),
        (EXP, 0),
        (ADD, 0),
    ): "mne_transcendental_f64",
    (
        (LOAD_VAR, 2),
        (LOAD_CONST, 0),
        (GT, 0),
        (LOAD_VAR, 0),
        (LOAD_VAR, 1),
        (MUL, 0),
        (LOAD_VAR, 2),
        (ADD, 0),
        (LOAD_VAR, 0),
        (LOAD_VAR, 1),
        (DIV, 0),
        (LOAD_VAR, 2),
        (SUB, 0),
        (WHERE, 0),
    ): "mne_conditional_f64",
}


class ExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class Program:
    expression: str
    input_names: tuple[str, ...]
    code: np.ndarray
    constants: np.ndarray
    result_bool: bool
    boolean_inputs: tuple[str, ...]
    specialized: str | None


class _Compiler:
    def __init__(self, expression: str):
        self.expression = expression
        self.instructions: list[tuple[int, int]] = []
        self.constants: list[float] = []
        self.names: set[str] = set()
        self.boolean_inputs: set[str] = set()
        self.depth = 0
        self.max_depth = 0

    def emit(self, opcode: int, argument: int = 0, delta: int = 0) -> None:
        self.instructions.append((opcode, argument))
        self.depth += delta
        self.max_depth = max(self.max_depth, self.depth)

    def compile(self) -> Program:
        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"invalid expression: {exc.msg}") from exc
        result_bool = self.visit(tree.body)
        if self.depth != 1:
            raise ExpressionError("invalid expression stack")
        if self.max_depth > 64:
            raise ExpressionError("expression requires more than 64 stack values")
        if len(self.instructions) > 256:
            raise ExpressionError("expression is too large (maximum 256 operations)")
        names = tuple(sorted(self.names))
        indexes = {name: index for index, name in enumerate(names)}
        code = [
            (opcode, indexes[argument] if opcode == LOAD_VAR else argument)
            for opcode, argument in self.instructions
        ]
        constants = self.constants or [0.0]
        return Program(
            self.expression,
            names,
            np.ascontiguousarray(code, dtype=np.int64),
            np.ascontiguousarray(constants, dtype=np.float64),
            result_bool,
            tuple(sorted(self.boolean_inputs)),
            _SPECIALIZED_CODES.get(tuple(code)),
        )

    def require_boolean(self, node: ast.AST, is_boolean: bool, message: str) -> None:
        if is_boolean:
            return
        if isinstance(node, ast.Name):
            self.boolean_inputs.add(node.id)
            return
        raise ExpressionError(message)

    def visit(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ExpressionError("names beginning with '_' are not allowed")
            self.names.add(node.id)
            self.instructions.append((LOAD_VAR, node.id))
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            return False
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (bool, int, float)):
                raise ExpressionError(
                    f"unsupported constant {type(node.value).__name__}"
                )
            if isinstance(node.value, int) and abs(node.value) > 2**53:
                raise ExpressionError(
                    "integer constants outside the exact float64 range are unsupported"
                )
            self.constants.append(float(node.value))
            self.emit(LOAD_CONST, len(self.constants) - 1, 1)
            return isinstance(node.value, bool)
        if isinstance(node, ast.BinOp):
            opcode = _BINARY.get(type(node.op))
            if opcode is None:
                raise ExpressionError(
                    f"unsupported binary operator {type(node.op).__name__}"
                )
            left_bool = self.visit(node.left)
            right_bool = self.visit(node.right)
            if opcode in (AND, OR):
                self.require_boolean(
                    node.left, left_bool, "'&' and '|' require boolean operands"
                )
                self.require_boolean(
                    node.right, right_bool, "'&' and '|' require boolean operands"
                )
            self.emit(opcode, delta=-1)
            return opcode in (AND, OR)
        if isinstance(node, ast.BoolOp):
            raise ExpressionError(
                "use bitwise '&' and '|' instead of Python 'and' and 'or'"
            )
        if isinstance(node, ast.UnaryOp):
            result_bool = self.visit(node.operand)
            if isinstance(node.op, ast.USub):
                self.emit(NEG)
                return False
            if isinstance(node.op, ast.UAdd):
                return result_bool
            if isinstance(node.op, (ast.Not, ast.Invert)):
                self.require_boolean(
                    node.operand,
                    result_bool,
                    "'not' and '~' require a boolean operand",
                )
                self.emit(NOT)
                return True
            raise ExpressionError(
                f"unsupported unary operator {type(node.op).__name__}"
            )
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ExpressionError("chained comparisons are not supported")
            opcode = _COMPARE.get(type(node.ops[0]))
            if opcode is None:
                raise ExpressionError(
                    f"unsupported comparison {type(node.ops[0]).__name__}"
                )
            self.visit(node.left)
            self.visit(node.comparators[0])
            self.emit(opcode, delta=-1)
            return True
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("only direct calls to supported functions are allowed")
            entry = _FUNCTIONS.get(node.func.id)
            if entry is None:
                raise ExpressionError(f"unsupported function {node.func.id!r}")
            opcode, arity = entry
            if len(node.args) != arity or node.keywords:
                raise ExpressionError(f"{node.func.id}() takes exactly {arity} arguments")
            kinds = [self.visit(argument) for argument in node.args]
            if opcode == WHERE:
                self.require_boolean(
                    node.args[0], kinds[0], "where() condition must be boolean"
                )
            self.emit(opcode, delta=1 - arity)
            return kinds[1] and kinds[2] if opcode == WHERE else False
        if isinstance(node, ast.IfExp):
            result_kind = self.visit(node.test)
            true_kind = self.visit(node.body)
            false_kind = self.visit(node.orelse)
            self.require_boolean(
                node.test, result_kind, "conditional test must be boolean"
            )
            self.emit(WHERE, delta=-2)
            return true_kind and false_kind
        raise ExpressionError(f"unsupported syntax {type(node).__name__}")


@lru_cache(maxsize=256)
def compile_expression(expression: str, sanitize: bool = True) -> Program:
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")
    if not sanitize:
        # This compiler never executes Python, so disabling the upstream name
        # sanitizer changes compatibility policy, not the safe AST whitelist.
        pass
    return _Compiler(expression).compile()


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in _SUPPORTED_DTYPES:
        if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize <= 8:
            array = array.astype(np.float64)
        else:
            raise TypeError(f"variable {name!r} has unsupported dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.signedinteger) and array.size:
        largest = np.max(array)
        smallest = np.min(array)
        if largest > 2**53 or smallest < -(2**53):
            raise ValueError(
                f"variable {name!r} contains integers outside the exact float64 range"
            )
    return np.ascontiguousarray(array)


def _metadata(
    arrays: list[np.ndarray], output_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ndim = len(output_shape)
    addresses = np.ascontiguousarray([addr(array) for array in arrays], dtype=np.int64)
    dtypes = np.ascontiguousarray(
        [_SUPPORTED_DTYPES[array.dtype] for array in arrays], dtype=np.int64
    )
    strides = np.zeros((len(arrays), ndim), dtype=np.int64)
    for row, array in enumerate(arrays):
        if array.shape == output_shape and array.flags.c_contiguous:
            strides[row, 0] = -1
            continue
        offset = ndim - array.ndim
        if offset < 0:
            raise ValueError("input has more dimensions than the broadcast result")
        for axis in range(array.ndim):
            output_axis = offset + axis
            size = array.shape[axis]
            target = output_shape[output_axis]
            if size not in (1, target):
                raise ValueError(
                    f"operands could not be broadcast to shape {output_shape}"
                )
            strides[row, output_axis] = (
                0 if size == 1 else array.strides[axis] // array.itemsize
            )
    return addresses, dtypes, np.ascontiguousarray(strides)


def _run_specialized(
    kernel,
    arrays: list[np.ndarray],
    result: np.ndarray,
    n: int,
    nthreads: int,
    worker_limit: int,
) -> None:
    workers = min(nthreads, worker_limit, os.cpu_count() or 1)
    if n < _PARALLEL_ELEMENTS or workers <= 1:
        kernel(
            addr(arrays[0]),
            addr(arrays[1]),
            addr(arrays[2]),
            addr(result),
            n,
            1,
        )
        return

    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
    elements_per_worker = (n // workers // 32) * 32
    futures = []
    for worker in range(workers):
        start = worker * elements_per_worker
        end = n if worker == workers - 1 else start + elements_per_worker
        futures.append(
            _pool.submit(
                kernel,
                addr(arrays[0]) + start * 8,
                addr(arrays[1]) + start * 8,
                addr(arrays[2]) + start * 8,
                addr(result) + start * result.itemsize,
                end - start,
                1,
            )
        )
    for future in futures:
        future.result()


def execute(
    program: Program,
    values: dict[str, Any],
    *,
    out: np.ndarray | None = None,
    order: str = "K",
    casting: str = "same_kind",
    nthreads: int = 1,
) -> np.ndarray:
    if order not in ("K", "C", "F", "A"):
        raise ValueError("order must be one of 'K', 'C', 'F', or 'A'")
    missing = [name for name in program.input_names if name not in values]
    if missing:
        raise KeyError(f"missing variable {missing[0]!r}")
    arrays = [_array(values[name], name) for name in program.input_names]
    for name, array in zip(program.input_names, arrays):
        if name in program.boolean_inputs and array.dtype != np.bool_:
            raise TypeError(f"variable {name!r} must have boolean dtype")
    try:
        logical_shape = np.broadcast_shapes(*(array.shape for array in arrays))
    except ValueError as exc:
        raise ValueError(f"operands could not be broadcast together: {exc}") from exc
    scalar_result = not logical_shape
    native_shape = logical_shape or (1,)
    if len(native_shape) > 8:
        raise ValueError("at most 8 broadcast dimensions are supported")
    dtype = np.bool_ if program.result_bool else np.float64
    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError("out must be a numpy.ndarray")
        if not out.flags.writeable:
            raise ValueError("out must be writable")
        if out.shape != logical_shape:
            raise ValueError(
                f"out has shape {out.shape}, expected broadcast shape {logical_shape}"
            )
    use_out_directly = (
        out is not None
        and out.dtype == dtype
        and out.flags.c_contiguous
    )
    result = out if use_out_directly else np.empty(native_shape, dtype=dtype, order="C")
    specialized = program.specialized
    can_specialize = (
        specialized is not None
        and not program.result_bool
        and order != "F"
        and all(array.dtype == np.float64 for array in arrays)
        and all(array.shape == logical_shape for array in arrays)
    )
    if specialized == "mne_polynomial_f64" and program.constants[0] != 0.25:
        can_specialize = False
    if specialized == "mne_conditional_f64" and program.constants[0] != 0.0:
        can_specialize = False
    if can_specialize:
        kernel = getattr(lib(), specialized)
        _run_specialized(
            kernel,
            arrays,
            result,
            math.prod(native_shape),
            nthreads,
            16,
        )
        if scalar_result:
            result = result.reshape(())
        if out is not None:
            if not use_out_directly:
                np.copyto(out, result, casting=casting)
            return out
        return result
    addresses, dtypes, strides = _metadata(arrays, native_shape)
    shape = np.ascontiguousarray(native_shape, dtype=np.int64)
    status = lib().mne_evaluate(
        addr(program.code),
        program.code.shape[0],
        addr(program.constants),
        addr(addresses),
        addr(dtypes),
        addr(strides),
        addr(shape),
        len(native_shape),
        addr(result),
        int(program.result_bool),
        math.prod(native_shape),
        nthreads,
    )
    if status:
        raise RuntimeError(f"Mojo evaluator failed with status {status}")
    if scalar_result:
        result = result.reshape(())
    if out is not None:
        if not use_out_directly:
            np.copyto(out, result, casting=casting)
        return out
    if order == "F" and result.ndim > 1:
        return np.array(result, order="F", copy=True)
    return result


def resolve_names(
    program: Program,
    local_dict: dict[str, Any] | None,
    global_dict: dict[str, Any] | None,
    frame_depth: int,
) -> dict[str, Any]:
    frame = inspect.currentframe()
    try:
        for _ in range(frame_depth):
            if frame is not None:
                frame = frame.f_back
        locals_source = local_dict if local_dict is not None else (
            frame.f_locals if frame is not None else {}
        )
        globals_source = global_dict if global_dict is not None else (
            frame.f_globals if frame is not None else {}
        )
        values = {}
        for name in program.input_names:
            if name in locals_source:
                values[name] = locals_source[name]
            elif name in globals_source:
                values[name] = globals_source[name]
            else:
                raise KeyError(name)
        return values
    finally:
        del frame


def default_threads() -> int:
    return min(16, os.cpu_count() or 1)
