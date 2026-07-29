# mojo-numexpr

`mojo-numexpr` evaluates NumExpr-style array expressions in compiled
[Mojo](https://www.modular.com/mojo). Python parses an expression once into a small,
safe bytecode program; Mojo then evaluates that program over whole NumPy arrays in one
fused SIMD pass, without allocating an intermediate array for every operator.

The public entry points follow upstream `numexpr`: `evaluate`, `re_evaluate`,
`NumExpr`, `set_num_threads`, `get_num_threads`, core/thread detection, and the VML
compatibility setters. The package is named `mojo_numexpr`, so it can be tested beside
the real upstream package rather than shadowing it.

## Covered subset

- Numeric arithmetic: `+`, `-`, `*`, `/`, `%`, `**`, unary `+`/`-`, and `abs`
- All six comparisons, boolean `&`, `|`, `~`, and boolean results
- `where`, conditional expressions, `minimum`, and `maximum`
- `sin`, `cos`, `tan`, inverse and hyperbolic variants, `exp`, `expm1`, `log`,
  `log1p`, `log10`, `sqrt`, `floor`, and `ceil`
- `float64`, `float32`, `int64`, `int32`, `uint8`, and boolean inputs
- Scalars, same-shaped arrays, and NumPy-compatible broadcasting through eight
  dimensions
- `local_dict`, `global_dict`, implicit caller locals, `out`, `order`, `casting`,
  reusable `NumExpr` objects, expression-plan caching, and `re_evaluate`

The parity suite compares supported shared behaviors directly with installed upstream
NumExpr 2.14.2, and checks the conditional-expression extension against NumPy. It
covers numerical results, broadcast shapes, boolean dtypes, output buffers, SIMD
tails, object calls, re-evaluation, thread settings, and rejected unsafe syntax.

This is intentionally not the complete NumExpr language. Reductions, complex numbers,
strings and `contains`, datetime values, attribute/subscript syntax, chained
comparisons, arbitrary NumExpr internals, and VML acceleration are not implemented.
Numeric results currently use a `float64` evaluation stack and return `float64`;
upstream preserves `float32` and integer result dtypes in some expressions. Integer
inputs and constants outside the exactly representable range `-2**53` through
`2**53` are rejected instead of being silently rounded. Other integer dtypes are
also rejected rather than silently narrowed. Boolean expressions do return `bool`.
Mojo's SIMD `pow` and logarithms can differ from upstream by a few parts in
`1e9`; the rest of the tested float64 surface agrees at roughly `1e-12`.

## Install

The repository pins the Mojo nightly it was tested with:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-numexpr.so`. Pixi adds `python/` to `PYTHONPATH`, so
an editable install is not required.

## Usage

```python
import numpy as np
import mojo_numexpr as ne

x = np.linspace(-3.0, 3.0, 1_000_000)
y = np.linspace(0.5, 1.5, 1_000_000)

result = ne.evaluate("where(x > 0, sin(x) + x*y, exp(x) - y)")
print(result.shape, result.dtype)
```

Explicit namespaces and reusable compilation use upstream's spelling:

```python
values = {"a": x, "b": y, "scale": 0.25}
result = ne.evaluate("a*b + scale", local_dict=values)

compiled = ne.NumExpr("a*b + c")
result = compiled(x, y, np.ones_like(x))  # arguments follow compiled.input_names
```

`out=` returns the supplied array and applies NumPy casting rules:

```python
target = np.empty_like(x)
assert ne.evaluate("x*x + 1", out=target) is target
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
x86-64 Linux 6.8.0 and glibc 2.39, using 16 threads and 5,000,000 `float64` elements.
Times are the best of five warmed calls and include result allocation. Ratio is
upstream NumExpr time divided by mojo-numexpr time, so a value below one means
mojo-numexpr is slower.

| expression | mojo-numexpr | numexpr 2.14.2 | NumPy | vs numexpr |
| --- | ---: | ---: | ---: | ---: |
| multiply-add | 17.57 ms | 15.89 ms | 65.39 ms | 0.90x slower |
| 8-op polynomial | 10.99 ms | 17.85 ms | 182.18 ms | 1.62x faster |
| transcendental | 33.82 ms | 25.25 ms | 403.76 ms | 0.75x slower |
| conditional | 11.10 ms | 11.99 ms | 167.86 ms | 1.08x faster |

The polynomial and conditional kernels are faster than upstream NumExpr on this run;
multiply-add and transcendental are slower. All four are faster than the equivalent
ordinary NumPy
expressions because the fused kernels do not materialize operator intermediates.
These are single-machine microbenchmarks, not general performance claims.

## How it works

`evaluate` parses Python expression syntax with `ast` and accepts only an explicit
operator/function whitelist. It never calls Python `eval`. The compiler lowers the
expression to postfix instructions, records variable names and constants, validates a
64-value maximum stack and a 256-instruction maximum program, then caches the plan.
The benchmark multiply-add, polynomial, transcendental, and conditional plans dispatch
to direct SIMD kernels. Other covered expressions use the general fused evaluator.

Python owns all memory. Inputs are made C-contiguous only when necessary and otherwise
cross the FFI without a copy. NumPy supplies output storage, and a compatible `out`
buffer is written directly without a temporary. ctypes passes buffer
addresses as 64-bit integers, explicit dtype codes, the output shape, and aligned
element strides to `@export(...) ... abi("C")` Mojo entry points. Mojo rebuilds
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside that wrapper, processes SIMD
chunks plus a scalar tail, and writes directly into the caller-owned output.

Large specialized calls are split across a reusable host thread pool; inputs below
1,048,576 elements stay serial. The general evaluator uses Mojo `parallelize` above
its 262,144-element threshold. Worker boundaries are aligned to SIMD chunks and only
the final worker handles the scalar remainder.

Broadcasting does not materialize expanded arrays. A same-shaped C-contiguous input is
loaded directly with SIMD; a scalar or broadcast dimension uses zero/aligned strides
to compute its source offset. Nothing allocated by Mojo crosses the ABI, so there is no
cross-runtime lifetime or deallocation protocol.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The Mojo implementation deliberately stays in one compilation unit because shared
library build cost is effectively fixed. Always run the benchmark through Pixi: its
task holds a machine-wide lock so concurrent jobs do not distort the results.

MIT licensed.
