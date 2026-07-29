"""Benchmark mojo-numexpr against upstream numexpr on identical expressions."""

from __future__ import annotations

import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import numexpr as upstream

import mojo_numexpr as mojo


def timeit(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    threads = min(
        int(os.environ.get("MNE_BENCH_THREADS", "16")),
        mojo.detect_number_of_cores(),
    )
    mojo.set_num_threads(threads)
    upstream.set_num_threads(threads)

    rng = np.random.default_rng(0)
    n = 5_000_000
    values = {
        "a": np.ascontiguousarray(rng.uniform(0.1, 1.1, n)),
        "b": np.ascontiguousarray(rng.uniform(0.2, 1.2, n)),
        "c": np.ascontiguousarray(rng.normal(size=n)),
    }
    cases = [
        ("multiply-add", "a*b+c", values, lambda: values["a"] * values["b"] + values["c"]),
        (
            "8-op polynomial",
            "a + b*c + a*a - b*b + c*c*0.25",
            values,
            lambda: (
                values["a"]
                + values["b"] * values["c"]
                + values["a"] * values["a"]
                - values["b"] * values["b"]
                + values["c"] * values["c"] * 0.25
            ),
        ),
        (
            "transcendental",
            "sin(a) + cos(b) + exp(-abs(c))",
            values,
            lambda: (
                np.sin(values["a"])
                + np.cos(values["b"])
                + np.exp(-np.abs(values["c"]))
            ),
        ),
        (
            "conditional",
            "where(c > 0, a*b + c, a/b - c)",
            values,
            lambda: np.where(
                values["c"] > 0,
                values["a"] * values["b"] + values["c"],
                values["a"] / values["b"] - values["c"],
            ),
        ),
    ]

    rows = []
    for name, expression, namespace, numpy_function in cases:
        mojo.evaluate(expression, local_dict=namespace)
        upstream.evaluate(expression, local_dict=namespace)
        numpy_function()
        mojo_time = timeit(
            lambda expression=expression, namespace=namespace: mojo.evaluate(
                expression, local_dict=namespace
            )
        )
        upstream_time = timeit(
            lambda expression=expression, namespace=namespace: upstream.evaluate(
                expression, local_dict=namespace
            )
        )
        numpy_time = timeit(numpy_function)
        rows.append(
            (
                name,
                mojo_time,
                upstream_time,
                numpy_time,
                upstream_time / mojo_time,
            )
        )

    cpu_model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    print(
        f"Machine: {cpu_model}, "
        f"{platform.platform()}, {threads} threads, {n:,} float64 elements"
    )
    print(f"Upstream NumExpr: {upstream.__version__}")
    print()
    print("| expression | mojo-numexpr | numexpr | NumPy | vs numexpr |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, mojo_time, upstream_time, numpy_time, ratio in rows:
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {numpy_time * 1e3:.2f} ms | "
            f"{ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
