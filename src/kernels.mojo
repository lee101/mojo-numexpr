"""Fused SIMD bytecode evaluator for NumExpr-style array expressions."""

from std.algorithm import parallelize
from std.math import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    ceil,
    cos,
    cosh,
    exp,
    expm1,
    floor,
    log,
    log10,
    log1p,
    pow,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime MAX_STACK = 64
comptime PARALLEL_ELEMENTS = 262_144


@export("mne_mul_add_f64")
def mne_mul_add_f64(
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    dst_addr: Int,
    n: Int,
    requested_workers: Int,
) abi("C"):
    var a = F64Ptr(unsafe_from_address=a_addr)
    var b = F64Ptr(unsafe_from_address=b_addr)
    var c = F64Ptr(unsafe_from_address=c_addr)
    var destination = F64Ptr(unsafe_from_address=dst_addr)
    var workers = min(requested_workers, num_physical_cores())
    if n < PARALLEL_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @parameter
    def process(worker: Int):
        var vectors = n // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = n
        var i = start
        while i + 4 * W <= end:
            destination.store(
                i, a.load[width=W](i) * b.load[width=W](i) + c.load[width=W](i)
            )
            destination.store(
                i + W,
                a.load[width=W](i + W) * b.load[width=W](i + W)
                + c.load[width=W](i + W),
            )
            destination.store(
                i + 2 * W,
                a.load[width=W](i + 2 * W) * b.load[width=W](i + 2 * W)
                + c.load[width=W](i + 2 * W),
            )
            destination.store(
                i + 3 * W,
                a.load[width=W](i + 3 * W) * b.load[width=W](i + 3 * W)
                + c.load[width=W](i + 3 * W),
            )
            i += 4 * W
        while i + W <= end:
            destination.store(
                i, a.load[width=W](i) * b.load[width=W](i) + c.load[width=W](i)
            )
            i += W
        while i < end:
            destination[i] = a[i] * b[i] + c[i]
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mne_polynomial_f64")
def mne_polynomial_f64(
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    dst_addr: Int,
    n: Int,
    requested_workers: Int,
) abi("C"):
    var a = F64Ptr(unsafe_from_address=a_addr)
    var b = F64Ptr(unsafe_from_address=b_addr)
    var c = F64Ptr(unsafe_from_address=c_addr)
    var destination = F64Ptr(unsafe_from_address=dst_addr)
    var workers = min(requested_workers, num_physical_cores())
    if n < PARALLEL_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @parameter
    def process(worker: Int):
        var vectors = n // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = n
        var i = start
        while i + W <= end:
            var av = a.load[width=W](i)
            var bv = b.load[width=W](i)
            var cv = c.load[width=W](i)
            destination.store(
                i, av + bv * cv + av * av - bv * bv + cv * cv * 0.25
            )
            i += W
        while i < end:
            destination[i] = (
                a[i]
                + b[i] * c[i]
                + a[i] * a[i]
                - b[i] * b[i]
                + c[i] * c[i] * 0.25
            )
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mne_transcendental_f64")
def mne_transcendental_f64(
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    dst_addr: Int,
    n: Int,
    requested_workers: Int,
) abi("C"):
    var a = F64Ptr(unsafe_from_address=a_addr)
    var b = F64Ptr(unsafe_from_address=b_addr)
    var c = F64Ptr(unsafe_from_address=c_addr)
    var destination = F64Ptr(unsafe_from_address=dst_addr)
    var workers = min(requested_workers, num_physical_cores())
    if n < PARALLEL_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @parameter
    def process(worker: Int):
        var vectors = n // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = n
        var i = start
        while i + W <= end:
            destination.store(
                i,
                sin(a.load[width=W](i))
                + cos(b.load[width=W](i))
                + exp(-abs(c.load[width=W](i))),
            )
            i += W
        while i < end:
            destination[i] = sin(a[i]) + cos(b[i]) + exp(-abs(c[i]))
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mne_conditional_f64")
def mne_conditional_f64(
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    dst_addr: Int,
    n: Int,
    requested_workers: Int,
) abi("C"):
    var a = F64Ptr(unsafe_from_address=a_addr)
    var b = F64Ptr(unsafe_from_address=b_addr)
    var c = F64Ptr(unsafe_from_address=c_addr)
    var destination = F64Ptr(unsafe_from_address=dst_addr)
    var workers = min(requested_workers, num_physical_cores())
    if n < PARALLEL_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @parameter
    def process(worker: Int):
        var vectors = n // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = n
        var i = start
        while i + W <= end:
            var av = a.load[width=W](i)
            var bv = b.load[width=W](i)
            var cv = c.load[width=W](i)
            destination.store(
                i, cv.gt(0.0).select(av * bv + cv, av / bv - cv)
            )
            i += W
        while i < end:
            if c[i] > 0.0:
                destination[i] = a[i] * b[i] + c[i]
            else:
                destination[i] = a[i] / b[i] - c[i]
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@always_inline
def load_input[width: Int](
    addresses: I64Ptr,
    dtypes: I64Ptr,
    strides: I64Ptr,
    shape: I64Ptr,
    ndim: Int,
    variable: Int,
    index: Int,
) -> SIMD[DType.float64, width]:
    var address = Int(addresses[variable])
    var dtype = Int(dtypes[variable])
    var stride_row = strides + variable * ndim
    if stride_row[0] == -1:
        if dtype == 0:
            return F64Ptr(unsafe_from_address=address).load[width=width](index)
        if dtype == 1:
            return F32Ptr(unsafe_from_address=address).load[width=width](index).cast[
                DType.float64
            ]()
        if dtype == 2:
            return I64Ptr(unsafe_from_address=address).load[width=width](index).cast[
                DType.float64
            ]()
        if dtype == 3:
            return I32Ptr(unsafe_from_address=address).load[width=width](index).cast[
                DType.float64
            ]()
        return U8Ptr(unsafe_from_address=address).load[width=width](index).cast[
            DType.float64
        ]()

    var values = SIMD[DType.float64, width](0.0)
    for lane in range(width):
        var flat = index + lane
        var offset = 0
        for rd in range(ndim):
            var d = ndim - 1 - rd
            var coordinate = flat % Int(shape[d])
            flat //= Int(shape[d])
            offset += coordinate * Int(stride_row[d])
        if dtype == 0:
            values[lane] = F64Ptr(unsafe_from_address=address)[offset]
        elif dtype == 1:
            values[lane] = Float64(F32Ptr(unsafe_from_address=address)[offset])
        elif dtype == 2:
            values[lane] = Float64(I64Ptr(unsafe_from_address=address)[offset])
        elif dtype == 3:
            values[lane] = Float64(I32Ptr(unsafe_from_address=address)[offset])
        else:
            values[lane] = Float64(U8Ptr(unsafe_from_address=address)[offset])
    return values


@always_inline
def execute_chunk[width: Int](
    code: I64Ptr,
    code_count: Int,
    constants: F64Ptr,
    addresses: I64Ptr,
    dtypes: I64Ptr,
    strides: I64Ptr,
    shape: I64Ptr,
    ndim: Int,
    index: Int,
) -> SIMD[DType.float64, width]:
    var stack = InlineArray[SIMD[DType.float64, width], MAX_STACK](
        fill=SIMD[DType.float64, width](0.0)
    )
    var sp = 0
    for pc in range(code_count):
        var op = Int(code[pc * 2])
        var argument = Int(code[pc * 2 + 1])
        if op == 1:
            stack[sp] = load_input[width](
                addresses, dtypes, strides, shape, ndim, argument, index
            )
            sp += 1
        elif op == 2:
            stack[sp] = SIMD[DType.float64, width](constants[argument])
            sp += 1
        elif op == 10:
            sp -= 1
            stack[sp - 1] += stack[sp]
        elif op == 11:
            sp -= 1
            stack[sp - 1] -= stack[sp]
        elif op == 12:
            sp -= 1
            stack[sp - 1] *= stack[sp]
        elif op == 13:
            sp -= 1
            stack[sp - 1] /= stack[sp]
        elif op == 14:
            sp -= 1
            stack[sp - 1] = pow(stack[sp - 1], stack[sp])
        elif op == 15:
            sp -= 1
            var quotient = floor(stack[sp - 1] / stack[sp])
            stack[sp - 1] -= quotient * stack[sp]
        elif op == 20:
            stack[sp - 1] = -stack[sp - 1]
        elif op == 21:
            stack[sp - 1] = abs(stack[sp - 1])
        elif op == 22:
            stack[sp - 1] = stack[sp - 1].eq(0.0).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 30:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].lt(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 31:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].le(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 32:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].gt(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 33:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].ge(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 34:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].eq(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 35:
            sp -= 1
            stack[sp - 1] = stack[sp - 1].ne(stack[sp]).select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 36:
            sp -= 1
            var both = stack[sp - 1].ne(0.0) & stack[sp].ne(0.0)
            stack[sp - 1] = both.select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 37:
            sp -= 1
            var either = stack[sp - 1].ne(0.0) | stack[sp].ne(0.0)
            stack[sp - 1] = either.select(
                SIMD[DType.float64, width](1.0),
                SIMD[DType.float64, width](0.0),
            )
        elif op == 40:
            stack[sp - 1] = sin(stack[sp - 1])
        elif op == 41:
            stack[sp - 1] = cos(stack[sp - 1])
        elif op == 42:
            stack[sp - 1] = tan(stack[sp - 1])
        elif op == 43:
            stack[sp - 1] = asin(stack[sp - 1])
        elif op == 44:
            stack[sp - 1] = acos(stack[sp - 1])
        elif op == 45:
            stack[sp - 1] = atan(stack[sp - 1])
        elif op == 46:
            sp -= 1
            stack[sp - 1] = atan2(stack[sp - 1], stack[sp])
        elif op == 47:
            stack[sp - 1] = sinh(stack[sp - 1])
        elif op == 48:
            stack[sp - 1] = cosh(stack[sp - 1])
        elif op == 49:
            stack[sp - 1] = tanh(stack[sp - 1])
        elif op == 50:
            stack[sp - 1] = asinh(stack[sp - 1])
        elif op == 51:
            stack[sp - 1] = acosh(stack[sp - 1])
        elif op == 52:
            stack[sp - 1] = atanh(stack[sp - 1])
        elif op == 53:
            stack[sp - 1] = exp(stack[sp - 1])
        elif op == 54:
            stack[sp - 1] = expm1(stack[sp - 1])
        elif op == 55:
            stack[sp - 1] = log(stack[sp - 1])
        elif op == 56:
            var value = stack[sp - 1]
            var square = value * value
            var series = (
                value
                - square * 0.5
                + square * value / 3.0
                - square * square * 0.25
                + square * square * value * 0.2
            )
            stack[sp - 1] = abs(value).lt(1.0e-4).select(
                series, log(SIMD[DType.float64, width](1.0) + value)
            )
        elif op == 57:
            stack[sp - 1] = log10(stack[sp - 1])
        elif op == 58:
            stack[sp - 1] = sqrt(stack[sp - 1])
        elif op == 59:
            stack[sp - 1] = floor(stack[sp - 1])
        elif op == 60:
            stack[sp - 1] = ceil(stack[sp - 1])
        elif op == 61:
            sp -= 1
            stack[sp - 1] = min(stack[sp - 1], stack[sp])
        elif op == 62:
            sp -= 1
            stack[sp - 1] = max(stack[sp - 1], stack[sp])
        elif op == 63:
            sp -= 2
            stack[sp - 1] = stack[sp - 1].ne(0.0).select(
                stack[sp], stack[sp + 1]
            )
    return stack[0]


@export("mne_evaluate")
def mne_evaluate(
    code_addr: Int,
    code_count: Int,
    constants_addr: Int,
    addresses_addr: Int,
    dtypes_addr: Int,
    strides_addr: Int,
    shape_addr: Int,
    ndim: Int,
    dst_addr: Int,
    result_bool: Int,
    n: Int,
    requested_workers: Int,
) abi("C") -> Int:
    if code_count < 1 or code_count > 256 or ndim < 1 or ndim > 8 or n < 0:
        return 1
    var code = I64Ptr(unsafe_from_address=code_addr)
    var constants = F64Ptr(unsafe_from_address=constants_addr)
    var addresses = I64Ptr(unsafe_from_address=addresses_addr)
    var dtypes = I64Ptr(unsafe_from_address=dtypes_addr)
    var strides = I64Ptr(unsafe_from_address=strides_addr)
    var shape = I64Ptr(unsafe_from_address=shape_addr)
    var workers = min(requested_workers, num_physical_cores())
    if n < PARALLEL_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @parameter
    def process(worker: Int):
        var vectors = n // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = n
        var i = start
        if result_bool == 0:
            var destination = F64Ptr(unsafe_from_address=dst_addr)
            while i + W <= end:
                destination.store(
                    i,
                    execute_chunk[W](
                        code,
                        code_count,
                        constants,
                        addresses,
                        dtypes,
                        strides,
                        shape,
                        ndim,
                        i,
                    ),
                )
                i += W
            while i < end:
                destination[i] = execute_chunk[1](
                    code,
                    code_count,
                    constants,
                    addresses,
                    dtypes,
                    strides,
                    shape,
                    ndim,
                    i,
                )[0]
                i += 1
        else:
            var destination = U8Ptr(unsafe_from_address=dst_addr)
            while i + W <= end:
                destination.store(
                    i,
                    execute_chunk[W](
                        code,
                        code_count,
                        constants,
                        addresses,
                        dtypes,
                        strides,
                        shape,
                        ndim,
                        i,
                    ).cast[DType.uint8](),
                )
                i += W
            while i < end:
                destination[i] = UInt8(
                    execute_chunk[1](
                        code,
                        code_count,
                        constants,
                        addresses,
                        dtypes,
                        strides,
                        shape,
                        ndim,
                        i,
                    )[0]
                    != 0.0
                )
                i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)
    return 0
