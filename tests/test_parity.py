"""Numerical and behavioral parity with upstream numexpr for the covered subset."""

from __future__ import annotations

import numpy as np
import numexpr as upstream
import pytest

import mojo_numexpr as ne
from mojo_numexpr import compiler
from mojo_numexpr.compiler import compile_expression


@pytest.fixture(scope="module")
def arrays():
    rng = np.random.default_rng(7)
    return {
        "a": np.ascontiguousarray(rng.uniform(0.2, 1.2, 4099)),
        "b": np.ascontiguousarray(rng.uniform(0.4, 1.5, 4099)),
        "c": np.ascontiguousarray(rng.normal(size=4099)),
    }


def parity(expression, values, *, rtol=1e-12, atol=1e-12):
    expected = upstream.evaluate(expression, local_dict=values)
    actual = ne.evaluate(expression, local_dict=values)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    if expected.dtype == np.bool_:
        assert actual.dtype == expected.dtype
        assert np.array_equal(actual, expected)
    return actual


def test_fused_arithmetic_parity(arrays):
    parity("a * b + c / 3.0 - 2.0", arrays)


@pytest.mark.parametrize(
    "expression",
    [
        "a + b",
        "a - b",
        "a * b",
        "a / b",
        "a ** b",
        "c % b",
        "-c",
        "+c",
        "abs(c)",
    ],
)
def test_arithmetic_operator_parity(expression, arrays):
    tolerance = 3e-9 if "**" in expression else 2e-12
    parity(expression, arrays, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "expression",
    [
        "sin(c)",
        "cos(c)",
        "tan(c / 4)",
        "arcsin(c / (abs(c) + 2))",
        "arccos(c / (abs(c) + 2))",
        "arctan(c)",
        "arctan2(c, b)",
        "sinh(c / 4)",
        "cosh(c / 4)",
        "tanh(c)",
        "arcsinh(c)",
        "arccosh(a + 1)",
        "arctanh(c / (abs(c) + 2))",
        "exp(c / 4)",
        "expm1(c / 4)",
        "log(a)",
        "log1p(a)",
        "log10(a)",
        "sqrt(a)",
        "floor(c)",
        "ceil(c)",
    ],
)
def test_math_function_parity(expression, arrays):
    tolerance = 3e-9 if expression.startswith(("log(", "log1p(")) else 3e-12
    parity(expression, arrays, rtol=tolerance, atol=tolerance)


def test_minimum_maximum_parity(arrays):
    parity("minimum(a, b) + maximum(b, c)", arrays)


def test_where_parity(arrays):
    parity("where(c >= 0, a * b, -a / b)", arrays)


@pytest.mark.parametrize("expression", ["a < b", "a <= b", "a > b", "a >= b", "a == b", "a != b"])
def test_comparison_parity(expression, arrays):
    parity(expression, arrays)


def test_boolean_expression_parity(arrays):
    parity("(a < b) & ((c > 0) | (a == b))", arrays)
    parity("~(a < b)", arrays)


def test_boolean_inputs_and_conditional_expression():
    mask = np.array([True, False, True, False])
    x = np.arange(4.0)
    parity("mask & (x > 1)", {"mask": mask, "x": x})
    actual = ne.evaluate(
        "x + 1 if mask else x - 1", local_dict={"mask": mask, "x": x}
    )
    assert np.array_equal(actual, np.where(mask, x + 1, x - 1))


def test_scalar_broadcast_parity(arrays):
    parity("scale * a + offset", {"a": arrays["a"], "scale": 1.75, "offset": -0.2})


def test_multidimensional_broadcast_parity():
    a = np.arange(15, dtype=np.float64).reshape(5, 3, 1)
    b = np.linspace(0.5, 2.0, 7).reshape(1, 1, 7)
    c = np.linspace(-1, 1, 3).reshape(1, 3, 1)
    parity("a * b + c", {"a": a, "b": b, "c": c})


def test_eight_dimensional_broadcast_boundary():
    a = np.arange(3.0).reshape((1,) * 7 + (3,))
    b = np.array(2.0)
    parity("a+b", {"a": a, "b": b})


def test_empty_and_simd_tail_shapes():
    for size in (0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33):
        x = np.linspace(0.1, 1.0, size)
        parity("sin(x) + x*x", {"x": x})


@pytest.mark.parametrize(
    "expression",
    [
        "a*b+c",
        "a + b*c + a*a - b*b + c*c*0.25",
        "sin(a) + cos(b) + exp(-abs(c))",
        "where(c > 0, a*b + c, a/b - c)",
    ],
)
def test_specialized_simd_tail(expression):
    size = 37
    values = {
        "a": np.linspace(0.1, 1.1, size),
        "b": np.linspace(0.2, 1.2, size),
        "c": np.linspace(-1.0, 1.0, size),
    }
    parity(expression, values)


def test_specialized_parallel_threshold():
    previous_threads = ne.get_num_threads()
    previous_pool = compiler._pool
    compiler._pool = None
    try:
        ne.set_num_threads(min(4, ne.detect_number_of_cores()))
        below = compiler._PARALLEL_ELEMENTS - 1
        below_values = {
            "a": np.linspace(0.1, 1.1, below),
            "b": np.linspace(0.2, 1.2, below),
            "c": np.linspace(-1.0, 1.0, below),
        }
        ne.evaluate(
            "a + b*c + a*a - b*b + c*c*0.25",
            local_dict=below_values,
        )
        assert compiler._pool is None

        size = compiler._PARALLEL_ELEMENTS + 17
        values = {
            "a": np.linspace(0.1, 1.1, size),
            "b": np.linspace(0.2, 1.2, size),
            "c": np.linspace(-1.0, 1.0, size),
        }
        parity("a + b*c + a*a - b*b + c*c*0.25", values)
        assert compiler._pool is not None
    finally:
        if compiler._pool is not None:
            compiler._pool.shutdown()
        compiler._pool = previous_pool
        ne.set_num_threads(previous_threads)


def test_float32_input_values():
    x = np.linspace(-2, 2, 1001, dtype=np.float32)
    parity("sin(x) + x*x", {"x": x}, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
def test_integer_input_values(dtype):
    x = np.arange(100, dtype=dtype)
    parity("x * 3 + 7", {"x": x})


def test_lossy_integer_inputs_and_constants_are_rejected():
    with pytest.raises(ValueError, match="exact float64 range"):
        ne.evaluate("x+1", local_dict={"x": np.array([2**53 + 1], dtype=np.int64)})
    with pytest.raises(TypeError, match="unsupported dtype"):
        ne.evaluate("x+1", local_dict={"x": np.array([1], dtype=np.uint64)})
    with pytest.raises(ne.ExpressionError, match="exact float64 range"):
        ne.evaluate("x+9007199254740993", local_dict={"x": np.array([1.0])})
    if hasattr(np, "float128"):
        with pytest.raises(TypeError, match="unsupported dtype"):
            ne.evaluate("x+1", local_dict={"x": np.array([1], dtype=np.float128)})


def test_out_parameter_identity_and_parity(arrays):
    expected = upstream.evaluate("a*b+c", local_dict=arrays)
    target = np.empty_like(expected)
    returned = ne.evaluate("a*b+c", local_dict=arrays, out=target)
    assert returned is target
    assert np.allclose(target, expected)


def test_out_casting_rules(arrays):
    target = np.empty(arrays["a"].shape, dtype=np.float32)
    ne.evaluate("a+b", local_dict=arrays, out=target, casting="unsafe")
    expected = upstream.evaluate("a+b", local_dict=arrays).astype(np.float32)
    assert np.array_equal(target, expected)
    with pytest.raises(TypeError):
        ne.evaluate("a+b", local_dict=arrays, out=target, casting="safe")
    target.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        ne.evaluate("a+b", local_dict=arrays, out=target, casting="unsafe")


@pytest.mark.parametrize("order", ["K", "C", "F", "A"])
def test_result_order(order):
    x = np.arange(24.0).reshape(4, 6)
    result = ne.evaluate("x+1", local_dict={"x": x}, order=order)
    if order == "F":
        assert result.flags.f_contiguous
    assert np.array_equal(
        result, upstream.evaluate("x+1", local_dict={"x": x}, order=order)
    )


def test_implicit_local_resolution():
    x = np.arange(20.0)
    assert np.array_equal(ne.evaluate("x * 2"), upstream.evaluate("x * 2"))


def test_explicit_global_resolution():
    x = np.arange(5.0)
    assert np.array_equal(
        ne.evaluate("x+1", local_dict={}, global_dict={"x": x}),
        upstream.evaluate("x+1", local_dict={"x": x}),
    )


def test_numexpr_object_parity(arrays):
    ours = ne.NumExpr("a*b+c")
    theirs = upstream.NumExpr("a*b+c")
    assert ours.input_names == theirs.input_names
    actual = ours(arrays["a"], arrays["b"], arrays["c"])
    expected = theirs(arrays["a"], arrays["b"], arrays["c"])
    assert np.allclose(actual, expected)


def test_re_evaluate_with_new_values():
    x = np.arange(10.0)
    ne.evaluate("x*x+1", local_dict={"x": x})
    new = np.arange(10.0) + 2
    actual = ne.re_evaluate(local_dict={"x": new})
    assert np.array_equal(actual, upstream.evaluate("x*x+1", local_dict={"x": new}))


def test_program_cache():
    assert compile_expression("x+1") is compile_expression("x+1")


def test_thread_controls(arrays):
    previous = ne.get_num_threads()
    try:
        assert ne.set_num_threads(1) == previous
        assert ne.get_num_threads() == 1
        one = ne.evaluate("sin(a)+cos(b)", local_dict=arrays)
        ne.set_num_threads(min(4, ne.detect_number_of_cores()))
        many = ne.evaluate("sin(a)+cos(b)", local_dict=arrays)
        assert np.array_equal(one, many)
    finally:
        ne.set_num_threads(previous)


def test_compatibility_helpers(capsys):
    assert ne.detect_number_of_cores() >= 1
    assert ne.detect_number_of_threads() >= 1
    assert ne.set_vml_num_threads(2) is None
    assert ne.set_vml_accuracy_mode("high") is None
    with pytest.raises(ValueError):
        ne.set_vml_accuracy_mode("invalid")
    ne.print_versions()
    assert "mojo-numexpr version" in capsys.readouterr().out


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "a.real",
        "open('x')",
        "[x for x in a]",
        "a[0]",
        "a // 2",
        "a < b < c",
        "a and b",
    ],
)
def test_unsupported_or_unsafe_syntax_is_rejected(expression):
    with pytest.raises(ne.ExpressionError):
        ne.evaluate(expression, local_dict={"a": np.arange(3.0), "b": np.arange(3.0), "c": np.arange(3.0)})


@pytest.mark.parametrize("expression", ["~a", "where(a, b, c)"])
def test_boolean_context_rejects_numeric_array(expression):
    with pytest.raises(TypeError, match="boolean dtype"):
        ne.evaluate(
            expression,
            local_dict={
                "a": np.arange(3.0),
                "b": np.arange(3.0),
                "c": np.arange(3.0),
            },
        )


def test_shape_and_missing_variable_errors():
    with pytest.raises(ValueError):
        ne.evaluate("a+b", local_dict={"a": np.zeros((2, 3)), "b": np.zeros((4,))})
    with pytest.raises(KeyError):
        ne.evaluate("a+b", local_dict={"a": np.zeros(3)})
