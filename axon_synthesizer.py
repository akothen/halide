from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from threading import Lock
from typing import Any, Callable, Optional, List

import z3


class _OpRef:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"nl.{self.name}"


class _NLNamespace:
    def __getattr__(self, name: str) -> _OpRef:
        op = _OpRef(name)
        setattr(self, name, op)
        return op


nl = _NLNamespace()


_ID_COUNTER = 1000
_BODY_COUNTER = 2000
_LARGE_POSITIVE_SENTINEL = 1e30
_BINARY_UFS: dict[str, z3.FuncDeclRef] = {}
_UNARY_UFS: dict[str, z3.FuncDeclRef] = {}
_COMPARE_UFS: dict[str, z3.FuncDeclRef] = {}
_ID_LOCK = Lock()
_BODY_LOCK = Lock()
_UF_LOCK = Lock()
_SEMANTICS_LOCK = Lock()


def _gen_id(prefix: Optional[str] = None) -> str:
    global _ID_COUNTER
    with _ID_LOCK:
        _ID_COUNTER += 1
        next_id = _ID_COUNTER
    if prefix is not None:
        return f"{prefix}_{next_id}"
    return f"id_{next_id}"


def _fresh_body_id() -> int:
    global _BODY_COUNTER
    with _BODY_LOCK:
        _BODY_COUNTER += 1
        return _BODY_COUNTER


def _tensor_function(name: str, rank: int) -> z3.FuncDeclRef:
    return z3.Function(name, *([z3.IntSort()] * builtins.max(1, rank)), z3.RealSort())


BODY1 = z3.Function("BODY1", z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.RealSort())
REDUCE1 = z3.Function("REDUCE1", z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.RealSort())
BODY2 = z3.Function("BODY2", z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.RealSort())
REDUCE2 = z3.Function("REDUCE2", z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.RealSort())

_POW_FN = z3.Function("NKI_POW", z3.RealSort(), z3.RealSort(), z3.RealSort())
_EXP_FN = z3.Function("NKI_EXP", z3.RealSort(), z3.RealSort())


@dataclass
class Context:
    facts: list[z3.BoolRef] = field(default_factory=list)

    def add(self, *facts: z3.BoolRef) -> None:
        self.facts.extend(facts)

    def extend(self, facts: list[z3.BoolRef]) -> None:
        self.facts.extend(facts)

    def merged(self, *others: "Context") -> "Context":
        out = Context(list(self.facts))
        for o in others:
            out.extend(o.facts)
        return out


@dataclass
class ShapeExpr:
    dims: list[z3.ArithRef]

    @property
    def rank(self) -> int:
        return len(self.dims)


@dataclass
class ShapeResult:
    out: ShapeExpr
    ctx: Context


@dataclass
class ReductionDesc:
    body_id: int
    extent: z3.ArithRef
    outer_rank: int


@dataclass
class Semantics:
    name: str
    shape: ShapeExpr
    fn: z3.FuncDeclRef
    ctx: Context = field(default_factory=Context)
    reduction: Optional[ReductionDesc] = None


@dataclass
class SymExpr:
    op: str
    inputs: list["SymExpr"]
    shape: tuple[z3.ArithRef, ...]
    attrs: dict[str, Any]
    name: str


class SymTensor:
    def __init__(
        self,
        id: str,
        shape: Optional[tuple[Any, ...]] = None,
        expr: Optional[SymExpr] = None,
        rank: Optional[int] = None,
    ):
        self.id = id
        if expr is not None:
            self.expr = expr
            self.shape = expr.shape
            return
        if shape is not None:
            self.shape = tuple(z3.IntVal(d) if isinstance(d, int) else d for d in shape)
        else:
            if rank is None:
                raise ValueError("Either shape or rank must be provided")
            self.shape = tuple(z3.Int(f"{id}_d{k}") for k in range(rank))
        self.expr = SymExpr("input", [], self.shape, {"shape": self.shape}, id)

    @property
    def rank(self) -> int:
        return len(self.shape)


_SEMANTICS: dict[str, tuple[
    Optional[Callable[[list[ShapeExpr], dict[str, Any]], ShapeResult]],
    Optional[Callable[[SymExpr, list[Semantics], ShapeExpr], Semantics]],
]] = {}


def semantics():
    def _decorator(fn):
        _SEMANTICS.setdefault(fn.__name__, (None, None))
        return fn

    return _decorator


def register_semantics(
    op_name: str,
    shape_rule_fn: Callable[[list[ShapeExpr], dict[str, Any]], ShapeResult],
    compile_rule_fn: Callable[[SymExpr, list[Semantics], ShapeExpr], Semantics],
) -> None:
    _SEMANTICS[op_name] = (shape_rule_fn, compile_rule_fn)


def _ensure_semantics(
    op_name: str,
    shape_rule_fn: Callable[[list[ShapeExpr], dict[str, Any]], ShapeResult],
    compile_rule_fn: Callable[[SymExpr, list[Semantics], ShapeExpr], Semantics],
) -> None:
    with _SEMANTICS_LOCK:
        current = _SEMANTICS.get(op_name)
        if current is None or current == (None, None):
            register_semantics(op_name, shape_rule_fn, compile_rule_fn)


def _to_dim(d: Any) -> z3.ArithRef:
    return z3.IntVal(d) if isinstance(d, int) else d


def _is_sym_tensor(x: Any) -> bool:
    return isinstance(x, SymTensor)


def _new_sym_tensor(op: str, inputs: list[SymTensor], attrs: dict[str, Any], out_shape: tuple[Any, ...]) -> SymTensor:
    out_id = attrs.get("name") or _gen_id(op)
    expr = SymExpr(op, [i.expr for i in inputs], tuple(_to_dim(d) for d in out_shape), attrs, out_id)
    return SymTensor(out_id, expr=expr)


def _public_pointwise_binary(x: Any, y: Any, op: _OpRef) -> Any:
    assert _is_sym_tensor(x) or _is_sym_tensor(y)
    if _is_sym_tensor(x) and _is_sym_tensor(y):
        return tensor_tensor(dst=None, data1=x, data2=y, op=op)
    if _is_sym_tensor(x):
        return tensor_scalar(dst=None, data=x, op0=op, operand0=y)
    return tensor_scalar(dst=None, data=y, op0=op, operand0=x, reverse0=True)


def _public_unary(op_name: str, x: Any, **attrs: Any) -> Any:
    assert _is_sym_tensor(x)
    return _new_sym_tensor(op_name, [x], attrs, x.shape)


def _public_binary(op_name: str, x: Any, y: Any, **attrs: Any) -> Any:
    assert _is_sym_tensor(x) or _is_sym_tensor(y)
    if _is_sym_tensor(x) and _is_sym_tensor(y):
        return _new_sym_tensor(op_name, [x, y], attrs, _public_broadcast_shape_tuple(x.shape, y.shape))
    if _is_sym_tensor(x):
        return _new_sym_tensor(op_name, [x], {"scalar": y, **attrs}, x.shape)
    return _new_sym_tensor(op_name, [y], {"scalar": x, "reverse": True, **attrs}, y.shape)


def _public_reduce(op_name: str, x: Any, axis: Any, *, keepdims: bool = False, **attrs: Any) -> Any:
    assert _is_sym_tensor(x)
    out_shape = _public_reduce_out_shape(x.shape, axis, keepdims)
    return _new_sym_tensor(op_name, [x], {"axis": axis, "keepdims": keepdims, **attrs}, out_shape)


def _public_dynamic_slice(start: Any, size: Any) -> Any:
    assert isinstance(start, int) and isinstance(size, int)
    return slice(start, start + size)


def _public_gather_flattened(data: Any, indices: Any, axis: Any) -> Any:
    assert (_is_sym_tensor(data) and _is_sym_tensor(indices))
    out_shape = (data.shape[0], *indices.shape[1:])
    return _new_sym_tensor("gather_flattened", [data, indices], {"axis": axis}, out_shape)


def _public_rms_norm(x: Any, w: Any) -> Any:
    assert _is_sym_tensor(x)
    inputs = [value for value in (x, w) if _is_sym_tensor(value)]
    return _new_sym_tensor("rms_norm", inputs, {}, x.shape)


def _public_store(dst: Any, value: Any) -> Any:
    assert _is_sym_tensor(value)
    out_shape = _default_out_shape(dst, value)
    return _new_sym_tensor("store", [value], {"out_shape": out_shape}, out_shape)


def _public_transpose_out_shape(x: SymTensor) -> tuple[Any, ...]:
    return (x.shape[1], x.shape[0]) if len(x.shape) == 2 else x.shape


def _arith_equal(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    lhs_expr = _to_dim(lhs) if isinstance(lhs, (int, z3.ArithRef)) else lhs
    rhs_expr = _to_dim(rhs) if isinstance(rhs, (int, z3.ArithRef)) else rhs
    if isinstance(lhs_expr, z3.ArithRef) and isinstance(rhs_expr, z3.ArithRef):
        return z3.is_true(z3.simplify(lhs_expr == rhs_expr))
    return lhs_expr == rhs_expr


def _broadcast_shape(a: ShapeExpr, b: ShapeExpr) -> ShapeResult:
    rank = builtins.max(a.rank, b.rank)
    ad = [z3.IntVal(1)] * (rank - a.rank) + list(a.dims)
    bd = [z3.IntVal(1)] * (rank - b.rank) + list(b.dims)
    out: list[z3.ArithRef] = []
    ctx = Context([d > 0 for d in ad + bd])
    for da, db in zip(ad, bd):
        ctx.add(z3.Or(da == db, da == 1, db == 1))
        out.append(z3.If(da == 1, db, da))
    return ShapeResult(ShapeExpr(out), ctx)


def _broadcast_indices(src: Semantics, out_shape: ShapeExpr, indices: list[z3.ArithRef]) -> list[z3.ArithRef]:
    src_dims = list(src.shape.dims)
    if len(indices) < out_shape.rank:
        aligned_indices = [z3.IntVal(0)] * (out_shape.rank - len(indices)) + list(indices)
    elif len(indices) > out_shape.rank:
        aligned_indices = list(indices[len(indices) - out_shape.rank :])
    else:
        aligned_indices = list(indices)
    src_to_out: list[Optional[int]] = [None] * len(src_dims)
    out_pos = out_shape.rank - 1
    for src_pos in range(len(src_dims) - 1, -1, -1):
        src_dim = src_dims[src_pos]
        if out_pos >= 0:
            src_to_out[src_pos] = out_pos
            out_pos -= 1
            continue
        # Preserve symbolic lowering progress by treating unmatched 
        # leading source dims as squeezable.
        # This is required for lifted affine-write updates where loop 
        # dimensions are external to update indices.
        src_to_out[src_pos] = None
    out: list[z3.ArithRef] = []
    for src_pos, src_dim in enumerate(src_dims):
        mapped_pos = src_to_out[src_pos]
        if mapped_pos is None:
            out.append(z3.IntVal(0))
        elif _arith_equal(src_dim, 1):
            out.append(z3.IntVal(0))
        else:
            out.append(aligned_indices[mapped_pos])
    return out


def _operand_to_expr(op: Any) -> Any:
    return op.name if isinstance(op, _OpRef) else op


def _apply_binary(op: Any, lhs: z3.ArithRef, rhs: z3.ArithRef) -> z3.ArithRef:
    opn = _operand_to_expr(op)
    if opn in ("add", "plus", "maximum", "max"):
        return lhs + rhs if opn in ("add", "plus") else z3.If(lhs >= rhs, lhs, rhs)
    if opn in ("subtract", "sub"):
        return lhs - rhs
    if opn in ("multiply", "mul"):
        return lhs * rhs
    if opn in ("divide", "div"):
        return lhs / rhs
    if opn in ("minimum", "min"):
        return z3.If(lhs <= rhs, lhs, rhs)
    if opn in ("equal",):
        return z3.If(lhs == rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("less",):
        return z3.If(lhs < rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("less_equal",):
        return z3.If(lhs <= rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("greater",):
        return z3.If(lhs > rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("greater_equal",):
        return z3.If(lhs >= rhs, z3.RealVal(1), z3.RealVal(0))
    if opn in ("power",):
        return _POW_FN(lhs, rhs)
    key = str(opn)
    with _UF_LOCK:
        if key not in _BINARY_UFS:
            _BINARY_UFS[key] = z3.Function(f"NKI_BIN_{len(_BINARY_UFS)}", z3.RealSort(), z3.RealSort(), z3.RealSort())
        fn = _BINARY_UFS[key]
    return fn(lhs, rhs)


def _apply_activation(op: Any, x: z3.ArithRef) -> tuple[z3.ArithRef, list[z3.BoolRef]]:
    opn = _operand_to_expr(op)
    if opn in ("copy", "identity", None):
        return x, []
    if opn in ("relu",):
        return z3.If(x >= 0, x, z3.RealVal(0)), []
    if opn in ("exp", "exponential"):
        y = _EXP_FN(x)
        return y, [y > 0, z3.Implies(x == 0, y == 1)]
    key = str(opn)
    with _UF_LOCK:
        if key not in _UNARY_UFS:
            _UNARY_UFS[key] = z3.Function(f"NKI_UN_{len(_UNARY_UFS)}", z3.RealSort(), z3.RealSort())
        fn = _UNARY_UFS[key]
    return fn(x), []


def _add_reduction_extensionality_axiom(solver: z3.Solver, lhs: Semantics, rhs: Semantics) -> None:
    lhs_red = lhs.reduction
    rhs_red = rhs.reduction
    if lhs_red is None or rhs_red is None or lhs_red.outer_rank != rhs_red.outer_rank:
        return
    lhs_bid = z3.IntVal(lhs_red.body_id)
    rhs_bid = z3.IntVal(rhs_red.body_id)
    if lhs_red.outer_rank == 1:
        i = z3.Int("equiv_reduce_i")
        k = z3.Int("equiv_reduce_k")
        same_body = z3.ForAll([k], BODY1(lhs_bid, i, k) == BODY1(rhs_bid, i, k))
        solver.add(
            z3.ForAll(
                [i],
                z3.Implies(
                    z3.And(lhs_red.extent == rhs_red.extent, same_body),
                    REDUCE1(lhs_bid, i, lhs_red.extent) == REDUCE1(rhs_bid, i, rhs_red.extent),
                ),
            )
        )
        return
    if lhs_red.outer_rank == 2:
        i = z3.Int("equiv_reduce_i")
        j = z3.Int("equiv_reduce_j")
        k = z3.Int("equiv_reduce_k")
        same_body = z3.ForAll([k], BODY2(lhs_bid, i, j, k) == BODY2(rhs_bid, i, j, k))
        solver.add(
            z3.ForAll(
                [i, j],
                z3.Implies(
                    z3.And(lhs_red.extent == rhs_red.extent, same_body),
                    REDUCE2(lhs_bid, i, j, lhs_red.extent) == REDUCE2(rhs_bid, i, j, rhs_red.extent),
                ),
            )
        )


def _default_out_shape(dst: Any, *srcs: Any) -> tuple[Any, ...]:
    if isinstance(dst, SymTensor):
        return dst.shape
    for s in srcs:
        if isinstance(s, SymTensor):
            return s.shape
    return (z3.IntVal(1),)


def _as_scalar(v: Any) -> z3.ArithRef:
    if isinstance(v, (int, float)):
        return z3.RealVal(v)
    if isinstance(v, z3.ArithRef):
        return v
    return z3.RealVal(0)


def _index_vars(name: str, rank: int, prefix: str = "i") -> list[z3.ArithRef]:
    return [z3.Int(f"{name}_{prefix}{k}") for k in range(rank)]


def _shape_ctx(*dims: z3.ArithRef) -> Context:
    return Context([d > 0 for d in dims])


def _shape_product(dims: list[z3.ArithRef]) -> z3.ArithRef:
    out: z3.ArithRef = z3.IntVal(1)
    for d in dims:
        out = out * d
    return out


def _int_floor_div(value: z3.ArithRef, divisor: int) -> z3.ArithRef:
    return value / z3.IntVal(divisor)


def _linear_index(dims: list[z3.ArithRef], indices: list[z3.ArithRef]) -> z3.ArithRef:
    out: z3.ArithRef = z3.IntVal(0)
    for d, idx in zip(dims, indices):
        out = out * d + idx
    return out


def _shape_from_out(out_shape: tuple[Any, ...]) -> ShapeResult:
    dims = [_to_dim(d) for d in out_shape]
    return ShapeResult(ShapeExpr(dims), _shape_ctx(*dims))


def _permute_dims(dims: list[z3.ArithRef], axes: list[int]) -> list[z3.ArithRef]:
    return [dims[a] for a in axes]


def _flattened_free_index(shape: ShapeExpr, indices: list[z3.ArithRef]) -> z3.ArithRef:
    if shape.rank <= 1:
        return z3.IntVal(0)
    return _linear_index(list(shape.dims[1:]), list(indices[1:]))


def _opaque_pointwise_semantics(
    expr: SymExpr,
    ins: list[Semantics],
    out_shape: ShapeExpr,
    arg_exprs: list[z3.ArithRef],
    *,
    extra_ctx: Optional[Context] = None,
    reduction: Optional[ReductionDesc] = None,
) -> Semantics:
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = Context()
    for sem in ins:
        ctx = ctx.merged(sem.ctx)
    if extra_ctx is not None:
        ctx = ctx.merged(extra_ctx)
    opaque = z3.Function(
        f"OPAQUE_{expr.name}",
        *([z3.RealSort()] * len(arg_exprs)),
        *([z3.IntSort()] * builtins.max(1, out_shape.rank)),
        z3.RealSort(),
    )
    opaque_args = [*arg_exprs, *(idx if idx else [z3.IntVal(0)])]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == opaque(*opaque_args)))
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _pattern_counts(pattern: Any) -> list[int]:
    counts: list[int] = []
    if pattern is None:
        return counts
    for pos, entry in enumerate(pattern):
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
            raise ValueError(f"pattern entries must be [step, count] pairs, got {entry!r} at position {pos} in {pattern!r}")
        counts.append(int(entry[1]))
    return counts


class dge_mode(Enum):
    r"""Descriptor Generation Engine mode."""
    unknown = 0
    """Unknown DGE mode, i.e., let compiler decide the DGE mode"""
    swdge = 1
    """Software DGE"""
    hwdge = 2
    """Hardware DGE"""
    none = 3
    """Not using DGE"""


class dma_engine(Enum):
    r"""DMA transfer engine.
        """
    dma = 1
    """Shared DMA with CoreBarrier synchronization (default). Can be triggered from any engine."""
    gpsimd_dma = 2
    """GPSIMD's internal DMA engine for low-latency SB-to-SB swaps in LNC=2.
        Implies GPSIMD as the trigger engine."""


class engine(Enum):
    r"""Neuron Device engines."""
    tensor = 1
    """Tensor Engine"""
    vector = 5
    """Vector Engine"""
    scalar = 2
    """Scalar Engine"""
    gpsimd = 3
    """GpSIMD Engine"""
    dma = 4
    """DMA Engine"""
    sync = 6
    """Sync Engine"""
    unknown = 0
    """Unknown Engine"""


class matmul_perf_mode(Enum):
    r"""Performance mode for matmul."""
    none = 'none'
    """Default mode, no performance optimization"""
    double_row = 'double_row'
    """Double FP8 mode, 2x matmul throughput by packing two FP8 weight/ifmap element pairs"""


class oob_mode(Enum):
    r"""Out-of-bounds access mode."""
    error = 0
    """Raise a runtime error when an out-of-bounds access is detected."""
    skip = 1
    """Silently skip the runtime out-of-bounds access."""


class reduce_cmd(Enum):
    r"""Engine register reduce commands."""
    idle = 0
    """Not using the accumulator registers"""
    reset = 1
    """Resets the accumulator registers to its initial state"""
    reduce = 2
    """Keeps accumulating over the current value of the accumulator registers"""
    reset_reduce = 3
    """Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators"""
    load_reduce = 4
    """Loads a value into the accumulator registers, then accumulate the results of the current instruction into the accumulators"""


class tile_size:
    r"""Hardware tile size constants (pmax, psum_fmax, gemm_stationary_fmax, etc.)"""
    bn_stats_fmax = ...
    """Maximum free dimension of BN_STATS"""
    gemm_moving_fmax = ...
    """Maximum free dimension of the moving operand of General Matrix Multiplication on Tensor Engine"""
    gemm_stationary_fmax = ...
    """Maximum free dimension of the stationary operand of General Matrix Multiplication on Tensor Engine"""
    pmax = ...
    """Maximum partition dimension of a tile"""
    psum_fmax = ...
    """Maximum free dimension of a tile on PSUM buffer"""
    psum_min_align = ...
    """Minimum byte alignment requirement for PSUM free dimension address"""
    sbuf_min_align = ...
    """Minimum byte alignment requirement for SBUF free dimension address"""
    total_available_sbuf_size = ...
    """Usable SBUF size per partition (total minus reserved bytes)."""



@semantics()
def abs(x, dtype=None):
    return _public_unary("abs", x)


@semantics()
def add(x, y, dtype=None):
    return _public_binary("add", x, y)


@semantics()
def all(x, axis, dtype=None):
    return _public_reduce("all", x, axis)


@semantics()
def arctan(x, dtype=None):
    return _public_unary("arctan", x)


@semantics()
def bitwise_and(x, y, dtype=None):
    return _public_binary("bitwise_and", x, y)


@semantics()
def bitwise_or(x, y, dtype=None):
    return _public_binary("bitwise_or", x, y)


@semantics()
def bitwise_xor(x, y, dtype=None):
    return _public_binary("bitwise_xor", x, y)


@semantics()
def broadcast_to(x, shape, dtype=None):
    assert _is_sym_tensor(x)
    return _new_sym_tensor("broadcast_to", [x], tuple(shape), out_shape=tuple(shape))


@semantics()
def ceil(x, dtype=None):
    return _public_unary("ceil", x)


@semantics()
def copy(x, dtype=None):
    return _public_unary("copy", x)


@semantics()
def cos(x, dtype=None):
    return _public_unary("cos", x)


@semantics()
def divide(x, y, dtype=None):
    return _public_binary("divide", x, y)


@semantics()
def dropout(x, rate, dtype=None):
    assert _is_sym_tensor(x)
    inputs = [x]
    if _is_sym_tensor(rate):
        inputs.append(rate)
    return _new_sym_tensor("dropout", inputs, {"rate": rate}, x.shape)


@semantics()
def ds(start, size):
    return _public_dynamic_slice(start, size)


@semantics()
def equal(x, y, dtype=None):
    return _public_binary("equal", x, y)


@semantics()
def erf(x, dtype=None):
    return _public_unary("erf", x)


@semantics()
def erf_dx(x, dtype=None):
    return _public_unary("erf_dx", x)


@semantics()
def exp(x, dtype=None):
    return _public_unary("exp", x)


@semantics()
def expand_dims(x, axis):
    return _new_sym_tensor("expand_dims", [x], {"axis": axis}, _public_expand_dims_out_shape(x.shape, int(axis))) if _is_sym_tensor(x) else ...


@semantics()
def floor(x, dtype=None):
    return _public_unary("floor", x)


@semantics()
def fmod(x, y, dtype=None):
    return _public_binary("fmod", x, y)


@semantics()
def gather_flattened(data, indices, axis=0, dtype=None):
    return _public_gather_flattened(data, indices, axis)


@semantics()
def gelu(x, dtype=None):
    return _public_unary("gelu", x)


@semantics()
def gelu_apprx_sigmoid(x, dtype=None):
    return _public_unary("gelu_apprx_sigmoid", x)


@semantics()
def gelu_apprx_sigmoid_dx(x, dtype=None):
    return _public_unary("gelu_apprx_sigmoid_dx", x)


@semantics()
def gelu_apprx_tanh(x, dtype=None):
    return _public_unary("gelu_apprx_tanh", x)


@semantics()
def gelu_dx(x, dtype=None):
    return _public_unary("gelu_dx", x)


@semantics()
def greater(x, y, dtype=None):
    return _public_binary("greater", x, y)


@semantics()
def greater_equal(x, y, dtype=None):
    return _public_binary("greater_equal", x, y)


@semantics()
def invert(x, dtype=None):
    return _public_unary("invert", x)


@semantics()
def left_shift(x, y, dtype=None):
    return _public_pointwise_binary(x, y, nl.left_shift)


@semantics()
def less(x, y, dtype=None):
    return _public_binary("less", x, y)


@semantics()
def less_equal(x, y, dtype=None):
    return _public_binary("less_equal", x, y)


@semantics()
def load(src, dtype=None):
    return dma_copy(dst=None, src=src)


@semantics()
def load_transpose2d(src, dtype=None):
    return dma_transpose(dst=None, src=src, axes=(1, 0))


@semantics()
def log(x, dtype=None):
    return _public_unary("log", x)


@semantics()
def logical_and(x, y, dtype=None):
    return _public_binary("logical_and", x, y)


@semantics()
def logical_not(x, dtype=None):
    return _public_unary("logical_not", x)


@semantics()
def logical_or(x, y, dtype=None):
    return _public_binary("logical_or", x, y)


@semantics()
def logical_xor(x, y, dtype=None):
    return _public_binary("logical_xor", x, y)


@semantics()
def matmul(x, y, transpose_x=False):
    stationary = x if transpose_x else nc_transpose(dst=None, data=x)
    assert (_is_sym_tensor(stationary) and _is_sym_tensor(y))
    return nc_matmul(dst=None, stationary=stationary, moving=y)


@semantics()
def max(x, axis, dtype=None, keepdims=False):
    return _public_reduce("max", x, axis, keepdims=keepdims)


@semantics()
def maximum(x, y, dtype=None):
    return _public_binary("maximum", x, y)


@semantics()
def mean(x, axis, dtype=None, keepdims=False):
    return _public_reduce("mean", x, axis, keepdims=keepdims)


@semantics()
def min(x, axis, dtype=None, keepdims=False):
    return _public_reduce("min", x, axis, keepdims=keepdims)


@semantics()
def minimum(x, y, dtype=None):
    return _public_binary("minimum", x, y)


@semantics()
def mish(x, dtype=None):
    r"""Mish activation, element-wise."""
    return activation(dst=None, op=nl.mish, data=x)


@semantics()
def mod(x, y, dtype=None):
    return _public_pointwise_binary(x, y, nl.mod)


@semantics()
def multiply(x, y, dtype=None):
    return _public_pointwise_binary(x, y, nl.multiply)


@semantics()
def negative(x, dtype=None):
    return _public_unary("negative", x)


@semantics()
def not_equal(x, y, dtype=None):
    return _public_binary("not_equal", x, y)


@semantics()
def power(x, y, dtype=None):
    return _public_binary("power", x, y)


@semantics()
def prod(x, axis, dtype=None, keepdims=False):
    return _public_reduce("prod", x, axis, keepdims=keepdims)


@semantics()
def reciprocal(x, dtype=None):
    return _public_unary("reciprocal", x)


@semantics()
def relu(x, dtype=None):
    return _public_unary("relu", x)


@semantics()
def right_shift(x, y, dtype=None):
    return _public_pointwise_binary(x, y, nl.right_shift)


@semantics()
def rms_norm(x, w, axis, n, epsilon=1e-06, dtype=None, compute_dtype=None):
    return _public_rms_norm(x, w)


@semantics()
def rsqrt(x, dtype=None):
    return _public_unary("rsqrt", x)


@semantics()
def sigmoid(x, dtype=None):
    return _public_unary("sigmoid", x)


@semantics()
def sign(x, dtype=None):
    return _public_unary("sign", x)


@semantics()
def silu(x, dtype=None):
    return _public_unary("silu", x)


@semantics()
def silu_dx(x, dtype=None):
    return _public_unary("silu_dx", x)


@semantics()
def sin(x, dtype=None):
    return _public_unary("sin", x)


@semantics()
def softmax(x, axis=-1, dtype=None):
    return _public_unary("softmax", x, axis=axis)


@semantics()
def softplus(x, dtype=None):
    return _public_unary("softplus", x)


@semantics()
def sqrt(x, dtype=None):
    return _public_unary("sqrt", x)


@semantics()
def square(x, dtype=None):
    return _public_unary("square", x)


@semantics()
def store(dst, value):
    return _public_store(dst, value)


@semantics()
def subtract(x, y, dtype=None):
    return _public_binary("subtract", x, y)


@semantics()
def sum(x, axis, dtype=None, keepdims=False):
    return _public_reduce("sum", x, axis, keepdims=keepdims)


@semantics()
def tan(x, dtype=None):
    return _public_unary("tan", x)


@semantics()
def tanh(x, dtype=None):
    return _public_unary("tanh", x)


@semantics()
def transpose(x, dtype=None):
    return _new_sym_tensor("transpose", [x], {}, _public_transpose_out_shape(x)) if _is_sym_tensor(x) else ...


@semantics()
def trunc(x, dtype=None):
    return _public_unary("trunc", x)


@semantics()
def var(x, axis, dtype=None, keepdims=False):
    return _public_reduce("var", x, axis, keepdims=keepdims)


@semantics()
def where(condition, x, y, dtype=None):
    assert (_is_sym_tensor(condition) and _is_sym_tensor(x) and _is_sym_tensor(y))
    xy_shape = _public_broadcast_shape_tuple(x.shape, y.shape)
    out_shape = _public_broadcast_shape_tuple(condition.shape, xy_shape)
    return _new_sym_tensor("where", [condition, x, y], {}, out_shape)


@semantics()
def activation(dst, op, data, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_activation(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_activation(expr, ins, out_shape)

    _ensure_semantics("activation", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {
        "op": op,
        "scale": scale,
        "reduce_op": reduce_op,
        "reduce_cmd": reduce_cmd,
        "name": name,
        "with_reduce": reduce_res is not None or reduce_op is not None,
    }
    if _is_sym_tensor(bias):
        attrs["bias_input_index"] = len(inputs)
        inputs.append(bias)
    else:
        attrs["bias_const"] = bias
    if _is_sym_tensor(scale):
        attrs["scale_input_index"] = len(inputs)
        inputs.append(scale)
    return _new_sym_tensor("activation", inputs, attrs, _default_out_shape(dst, data))


@semantics()
def activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_activation(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_activation_reduce(expr, ins, out_shape)

    _ensure_semantics("activation_reduce", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {"op": op, "reduce_op": reduce_op, "scale": scale, "name": name}
    if _is_sym_tensor(bias):
        attrs["bias_input_index"] = len(inputs)
        inputs.append(bias)
    else:
        attrs["bias_const"] = bias
    if _is_sym_tensor(scale):
        attrs["scale_input_index"] = len(inputs)
        inputs.append(scale)
    return _new_sym_tensor("activation_reduce", inputs, attrs, _default_out_shape(dst, data))


@semantics()
def affine_select(dst, pattern, channel_multiplier, on_true_tile, on_false_value, cmp_op=nl.equal, offset=0, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        pattern_counts = _pattern_counts(attrs.get("pattern", []))
        if data_shape.rank > 1 and pattern_counts:
            ctx.add(_shape_product(list(data_shape.dims[1:])) == _shape_product([z3.IntVal(n) for n in pattern_counts]))
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        on_true_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = on_true_sem.ctx.merged()
        flat_idx = _flattened_free_index(out_shape, idx)
        affine_val = z3.ToReal(z3.IntVal(int(expr.attrs.get("offset", 0)))) + z3.ToReal(idx[0] * z3.IntVal(int(expr.attrs.get("channel_multiplier", 0))) + flat_idx)
        pred = _compare_bool(expr.attrs.get("cmp_op"), affine_val, z3.RealVal(0))
        true_val = _call_broadcasted(on_true_sem, out_shape, idx)
        false_val = _as_scalar(expr.attrs.get("on_false_value"))
        ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(pred, true_val, false_val)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("affine_select", shape_rule, value_rule)

    assert _is_sym_tensor(on_true_tile)
    return _new_sym_tensor(
        "affine_select",
        [on_true_tile],
        {
            "pattern": pattern,
            "channel_multiplier": channel_multiplier,
            "on_false_value": on_false_value,
            "cmp_op": cmp_op,
            "offset": offset,
            "name": name,
        },
        _default_out_shape(dst, on_true_tile),
    )


@semantics()
def bn_aggr(dst, data, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        dims = [data_shape.dims[0], z3.IntVal(2)]
        ctx = _shape_ctx(*data_shape.dims)
        if data_shape.rank >= 2:
            ctx.add(data_shape.dims[1] % 3 == 0)
        return ShapeResult(ShapeExpr(dims), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=_shape_ctx(*out_shape.dims))

    _ensure_semantics("bn_aggr", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    out_shape = _default_out_shape(dst, data)
    if out_shape == data.shape:
        out_shape = (data.shape[0], z3.IntVal(2))
    return _new_sym_tensor("bn_aggr", [data], {"name": name}, out_shape)


@semantics()
def bn_stats(dst, data, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        return ShapeResult(ShapeExpr([data_shape.dims[0], z3.IntVal(6)]), _shape_ctx(*data_shape.dims))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=_shape_ctx(*out_shape.dims))

    _ensure_semantics("bn_stats", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    out_shape = _default_out_shape(dst, data)
    if out_shape == data.shape:
        out_shape = (data.shape[0], z3.IntVal(6))
    return _new_sym_tensor("bn_stats", [data], {"name": name}, out_shape)


@semantics()
def dma_compute(dst, srcs, reduce_op, scales=None, unique_indices=True, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        out_shape = attrs.get("out_shape")
        if out_shape is not None:
            return _shape_from_out(out_shape)
        if ins:
            first = ins[0]
            return ShapeResult(ShapeExpr(list(first.dims)), _shape_ctx(*first.dims))
        return _shape_from_out((z3.IntVal(1),))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = Context()
        for sem in ins:
            ctx = ctx.merged(sem.ctx)
        values = [_call_broadcasted(sem, out_shape, idx) for sem in ins]
        tmp = z3.RealVal(0)
        for value, scale in zip(values, expr.attrs.get("scales", [])):
            tmp = _apply_binary(expr.attrs.get("reduce_op"), tmp, value * _as_scalar(scale))
        ctx.add(z3.ForAll(idx, out_fn(*idx) == tmp))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("dma_compute", shape_rule, value_rule)

    sym_srcs = [src for src in srcs if _is_sym_tensor(src)]
    assert sym_srcs
    resolved_scales = list(scales) if scales is not None else [1.0] * len(srcs)
    return _new_sym_tensor(
        "dma_compute",
        sym_srcs,
        {
            "reduce_op": reduce_op,
            "scales": resolved_scales,
            "unique_indices": unique_indices,
            "out_shape": _default_out_shape(dst, *sym_srcs),
            "name": name,
        },
        _default_out_shape(dst, *sym_srcs),
    )


@semantics()
def dma_copy(dst, src, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown, engine=engine.unknown, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_same_as_first(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_copy(expr, ins, out_shape)

    _ensure_semantics("dma_copy", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    return _new_sym_tensor(
        "dma_copy",
        [src],
        {"oob_mode": oob_mode, "dge_mode": dge_mode, "engine": engine, "name": name},
        _default_out_shape(dst, src),
    )


@semantics()
def dma_transpose(dst, src, axes=None, dge_mode=dge_mode.unknown, oob_mode=oob_mode.error, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        src_shape = ins[0]
        axes_attr = attrs.get("axes")
        axes_list = list(axes_attr) if axes_attr is not None else (
            [1, 0] if src_shape.rank == 2 else [2, 1, 0] if src_shape.rank == 3 else [3, 1, 2, 0]
        )
        dims = _permute_dims(list(src_shape.dims), axes_list)
        return ShapeResult(ShapeExpr(dims), _shape_ctx(*src_shape.dims, *dims))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        src_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        axes_attr = expr.attrs.get("axes")
        axes_list = list(axes_attr) if axes_attr is not None else (
            [1, 0] if out_shape.rank == 2 else [2, 1, 0] if out_shape.rank == 3 else [3, 1, 2, 0]
        )
        src_idx = [idx[axes_list.index(axis)] for axis in range(len(axes_list))]
        ctx = src_sem.ctx.merged()
        ctx.add(z3.ForAll(idx, out_fn(*idx) == src_sem.fn(*src_idx)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("dma_transpose", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    resolved_axes = tuple(axes) if axes is not None else None
    out_shape = _default_out_shape(dst, src)
    if resolved_axes is not None:
        out_shape = tuple(out_shape[a] for a in resolved_axes)
    elif len(out_shape) == 2:
        out_shape = (out_shape[1], out_shape[0])
    elif len(out_shape) == 3:
        out_shape = (out_shape[2], out_shape[1], out_shape[0])
    elif len(out_shape) == 4:
        out_shape = (out_shape[3], out_shape[1], out_shape[2], out_shape[0])
    return _new_sym_tensor(
        "dma_transpose",
        [src],
        {"axes": resolved_axes, "dge_mode": dge_mode, "oob_mode": oob_mode, "name": name},
        out_shape,
    )


@semantics()
def dropout(dst, data, prob, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        if len(ins) > 1:
            ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        data_sem = ins[0]
        idx = _index_vars(expr.name, out_shape.rank)
        prob_value = _operand_value(expr.attrs, "prob_const", "prob_input_index", ins, out_shape, idx)
        data_value = _call_broadcasted(data_sem, out_shape, idx)
        drop_mask = z3.Function(
            f"DROP_MASK_{expr.name}",
            z3.RealSort(),
            *([z3.IntSort()] * builtins.max(1, out_shape.rank)),
            z3.BoolSort(),
        )
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = data_sem.ctx.merged(*[s.ctx for s in ins[1:]])
        mask_args = [prob_value, *(idx if idx else [z3.IntVal(0)])]
        value = z3.If(prob_value <= 0, data_value, z3.If(prob_value >= 1, z3.RealVal(0), z3.If(drop_mask(*mask_args), z3.RealVal(0), data_value)))
        ctx.add(z3.ForAll(idx, out_fn(*idx) == value))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("dropout", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {"name": name}
    if _is_sym_tensor(prob):
        attrs["prob_input_index"] = len(inputs)
        inputs.append(prob)
    else:
        attrs["prob_const"] = prob
    return _new_sym_tensor("dropout", inputs, attrs, _default_out_shape(dst, data))


@semantics()
def exponential(dst, src, max_value=0.0, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_init=0.0, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_activation(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_exponential(expr, ins, out_shape)

    _ensure_semantics("exponential", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    inputs = [src]
    attrs: dict[str, Any] = {
        "max_value": max_value,
        "reduce_cmd": reduce_cmd,
        "reduce_init": reduce_init,
        "with_reduce": reduce_res is not None,
        "name": name,
    }
    if _is_sym_tensor(max_value):
        attrs["max_input_index"] = len(inputs)
        inputs.append(max_value)
    return _new_sym_tensor("exponential", inputs, attrs, _default_out_shape(dst, src))


@semantics()
def iota(dst, pattern, offset=0, channel_multiplier=0, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        out_shape = attrs.get("out_shape", ())
        dims = [_to_dim(d) for d in out_shape]
        return ShapeResult(ShapeExpr(dims), Context([d > 0 for d in dims]))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = Context([d > 0 for d in out_shape.dims])
        if out_shape.rank != 2:
            ctx.add(z3.BoolVal(False))
            return Semantics(expr.name, out_shape, out_fn, ctx)
        i = z3.Int(f"{expr.name}_i")
        j = z3.Int(f"{expr.name}_j")
        offset = int(expr.attrs.get("offset", 0))
        channel_multiplier = int(expr.attrs.get("channel_multiplier", 0))
        step = _extract_iota_step(expr.attrs.get("pattern", []))
        val = z3.ToReal(
            offset
            + i * channel_multiplier
            + j * int(step)
        )
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == val))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics(
        "iota",
        shape_rule,
        value_rule,
    )

    out_shape = _default_out_shape(dst)
    assert out_shape
    return _new_sym_tensor(
        "iota",
        [],
        {"pattern": pattern, "offset": offset, "channel_multiplier": channel_multiplier, "out_shape": out_shape, "name": name},
        out_shape,
    )


@semantics()
def local_gather(dst, src_buffer, index, num_elem_per_idx=1, num_valid_indices=None, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_from_out(attrs.get("out_shape", tuple(ins[0].dims)))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=_shape_ctx(*out_shape.dims))

    _ensure_semantics("local_gather", shape_rule, value_rule)

    assert (_is_sym_tensor(src_buffer) and _is_sym_tensor(index))
    return _new_sym_tensor(
        "local_gather",
        [src_buffer, index],
        {"num_elem_per_idx": num_elem_per_idx, "num_valid_indices": num_valid_indices, "out_shape": _default_out_shape(dst, src_buffer), "name": name},
        _default_out_shape(dst, src_buffer),
    )


@semantics()
def max8(dst, src, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        src_shape = ins[0]
        return ShapeResult(ShapeExpr([src_shape.dims[0], z3.IntVal(8)]), _shape_ctx(*src_shape.dims))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=_shape_ctx(*out_shape.dims))

    _ensure_semantics("max8", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    out_shape = _default_out_shape(dst, src)
    if out_shape == src.shape:
        out_shape = (src.shape[0], z3.IntVal(8))
    return _new_sym_tensor("max8", [src], {"name": name}, out_shape)


@semantics()
def memset(dst, value, engine=engine.unknown, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_from_out(attrs.get("out_shape", (z3.IntVal(1),)))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = _shape_ctx(*out_shape.dims)
        ctx.add(z3.ForAll(idx, out_fn(*idx) == _as_scalar(expr.attrs.get("value"))))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("memset", shape_rule, value_rule)

    out_shape = _default_out_shape(dst)
    assert out_shape
    return _new_sym_tensor("memset", [], {"value": value, "engine": engine, "out_shape": out_shape, "name": name}, out_shape)


@semantics()
def nc_find_index8(dst, data, vals, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims, *ins[1].dims)
        return ShapeResult(ShapeExpr([data_shape.dims[0], z3.IntVal(8)]), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=_shape_ctx(*out_shape.dims))

    _ensure_semantics("nc_find_index8", shape_rule, value_rule)

    assert (_is_sym_tensor(data) and _is_sym_tensor(vals))
    out_shape = _default_out_shape(dst, data)
    if out_shape == data.shape:
        out_shape = (data.shape[0], z3.IntVal(8))
    return _new_sym_tensor("nc_find_index8", [data, vals], {"name": name}, out_shape)


@semantics()
def nc_match_replace8(dst, data, vals, imm, dst_idx=None, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims, *ins[1].dims)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _opaque_pointwise_semantics(expr, ins, out_shape, [_as_scalar(expr.attrs.get("imm"))], extra_ctx=_shape_ctx(*out_shape.dims))

    _ensure_semantics("nc_match_replace8", shape_rule, value_rule)

    assert (_is_sym_tensor(data) and _is_sym_tensor(vals))
    return _new_sym_tensor("nc_match_replace8", [data, vals], {"imm": imm, "name": name}, _default_out_shape(dst, data))


@semantics()
def nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=matmul_perf_mode.none, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        lhs, rhs = ins
        out_shape = attrs.get("out_shape")
        if out_shape is not None:
            return _shape_from_out(out_shape)
        if lhs.rank >= 2 and rhs.rank >= 2:
            dims = [lhs.dims[-1], rhs.dims[-1]]
            return ShapeResult(ShapeExpr(dims), _shape_ctx(*lhs.dims, *rhs.dims, *dims))
        return ShapeResult(ShapeExpr([lhs.dims[0], rhs.dims[-1] if rhs.rank else z3.IntVal(1)]), _shape_ctx(*lhs.dims, *rhs.dims))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        stationary_sem, moving_sem = ins
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = stationary_sem.ctx.merged(moving_sem.ctx)
        if stationary_sem.shape.rank == 2 and moving_sem.shape.rank == 2 and out_shape.rank == 2:
            m = z3.Int(f"{expr.name}_m")
            n = z3.Int(f"{expr.name}_n")
            k = z3.Int(f"{expr.name}_k")
            body_id = _fresh_body_id()
            ctx.add(z3.ForAll([m, n, k], BODY2(z3.IntVal(body_id), m, n, k) == stationary_sem.fn(k, m) * moving_sem.fn(k, n)))
            extent = stationary_sem.shape.dims[0]
            val = REDUCE2(z3.IntVal(body_id), m, n, extent)
            if expr.attrs.get("accumulate") is True:
                acc = z3.Function(f"ACC_{expr.name}", z3.IntSort(), z3.IntSort(), z3.RealSort())
                val = acc(m, n) + val
            ctx.add(z3.ForAll([m, n], out_fn(m, n) == val))
            return Semantics(expr.name, out_shape, out_fn, ctx, reduction=ReductionDesc(body_id, extent, outer_rank=2))
        return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=ctx)

    _ensure_semantics("nc_matmul", shape_rule, value_rule)

    assert (_is_sym_tensor(stationary) and _is_sym_tensor(moving))
    out_shape = _default_out_shape(dst, stationary)
    if out_shape == stationary.shape and stationary.rank >= 2 and moving.rank >= 2:
        out_shape = (stationary.shape[-1], moving.shape[-1])
    return _new_sym_tensor(
        "nc_matmul",
        [stationary, moving],
        {
            "is_stationary_onezero": is_stationary_onezero,
            "is_moving_onezero": is_moving_onezero,
            "is_transpose": is_transpose,
            "accumulate": accumulate,
            "tile_position": tile_position,
            "tile_size": tile_size,
            "perf_mode": perf_mode,
            "out_shape": out_shape,
            "name": name,
        },
        out_shape,
    )


@semantics()
def nc_n_gather(dst, data, indices, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape, idx_shape = ins
        dims = [data_shape.dims[0], *idx_shape.dims[1:]]
        return ShapeResult(ShapeExpr(dims), _shape_ctx(*data_shape.dims, *idx_shape.dims))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        data_sem, idx_sem = ins
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = data_sem.ctx.merged(idx_sem.ctx)
        flat = z3.ToInt(idx_sem.fn(idx[0], *idx[1:])) if out_shape.rank > 1 else z3.ToInt(idx_sem.fn(idx[0]))
        free_extent = _shape_product(list(data_sem.shape.dims[1:])) if data_sem.shape.rank > 1 else z3.IntVal(1)
        src_free = flat % free_extent
        if data_sem.shape.rank == 2:
            ctx.add(z3.ForAll(idx, out_fn(*idx) == data_sem.fn(idx[0], src_free)))
        else:
            opaque_src = z3.Function(f"GATHER_{expr.name}", z3.IntSort(), z3.IntSort(), z3.RealSort())
            ctx.add(z3.ForAll(idx, out_fn(*idx) == opaque_src(idx[0], src_free)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("nc_n_gather", shape_rule, value_rule)

    assert (_is_sym_tensor(data) and _is_sym_tensor(indices))
    out_shape = _default_out_shape(dst, indices)
    if out_shape == indices.shape:
        out_shape = (data.shape[0], *indices.shape[1:])
    return _new_sym_tensor("nc_n_gather", [data, indices], {"name": name}, out_shape)


@semantics()
def nc_stream_shuffle(dst, src, shuffle_mask, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        src_shape = ins[0]
        return ShapeResult(ShapeExpr(list(src_shape.dims)), _shape_ctx(*src_shape.dims))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        src_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = src_sem.ctx.merged()
        mask = expr.attrs.get("shuffle_mask", [])
        if len(mask) != 32 or out_shape.rank < 1:
            return _opaque_pointwise_semantics(expr, ins, out_shape, [], extra_ctx=ctx)
        part = idx[0]
        quadrant_base = part - (part % z3.IntVal(32))
        source_part = z3.IntVal(0)
        value = z3.RealVal(0)
        for pos, source in enumerate(mask):
            src_pos = part if int(source) == 255 else quadrant_base + z3.IntVal(int(source))
            source_part = z3.If((part % z3.IntVal(32)) == pos, src_pos, source_part)
        src_idx = [source_part, *idx[1:]]
        value = src_sem.fn(*src_idx) if out_shape.rank > 1 else src_sem.fn(source_part)
        ctx.add(z3.ForAll(idx, out_fn(*idx) == value))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("nc_stream_shuffle", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    return _new_sym_tensor("nc_stream_shuffle", [src], {"shuffle_mask": list(shuffle_mask), "name": name}, _default_out_shape(dst, src))


@semantics()
def nc_transpose(dst, data, engine=engine.unknown, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_nc_transpose(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_nc_transpose(expr, ins, out_shape)

    _ensure_semantics("nc_transpose", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    out_shape = _default_out_shape(dst, data)
    if len(out_shape) >= 2:
        out_shape = (out_shape[1], out_shape[0], *out_shape[2:])
    return _new_sym_tensor("nc_transpose", [data], {"engine": engine, "name": name}, out_shape)


@semantics()
def reciprocal(dst, data, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_unary_same(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_reciprocal(expr, ins, out_shape)

    _ensure_semantics("reciprocal", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    return _new_sym_tensor("reciprocal", [data], {"name": name}, _default_out_shape(dst, data))


@semantics()
def scalar_tensor_tensor(dst, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        if len(ins) > 1:
            ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
        if len(ins) > 2:
            ctx.extend(_broadcast_shape(data_shape, ins[2]).ctx.facts)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        data_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = data_sem.ctx.merged(*[s.ctx for s in ins[1:]])
        data_value = _call_broadcasted(data_sem, out_shape, idx)
        op0_value = _operand_value(expr.attrs, "operand0_const", "operand0_input_index", ins, out_shape, idx)
        lhs0, rhs0 = (op0_value, data_value) if expr.attrs.get("reverse0", False) else (data_value, op0_value)
        tmp = _apply_binary(expr.attrs.get("op0"), lhs0, rhs0)
        op1_value = _call_broadcasted(ins[int(expr.attrs["operand1_input_index"])], out_shape, idx)
        lhs1, rhs1 = (op1_value, tmp) if expr.attrs.get("reverse1", False) else (tmp, op1_value)
        ctx.add(z3.ForAll(idx, out_fn(*idx) == _apply_binary(expr.attrs.get("op1"), lhs1, rhs1)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("scalar_tensor_tensor", shape_rule, value_rule)

    assert (_is_sym_tensor(data) and _is_sym_tensor(operand1))
    inputs = [data]
    attrs: dict[str, Any] = {"op0": op0, "op1": op1, "reverse0": reverse0, "reverse1": reverse1, "name": name}
    if _is_sym_tensor(operand0):
        attrs["operand0_input_index"] = len(inputs)
        inputs.append(operand0)
    else:
        attrs["operand0_const"] = operand0
    attrs["operand1_input_index"] = len(inputs)
    inputs.append(operand1)
    return _new_sym_tensor("scalar_tensor_tensor", inputs, attrs, _default_out_shape(dst, data))


@semantics()
def select_reduce(dst, predicate, on_true, on_false, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_op=nl.maximum, reverse_pred=False, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_tensor_scalar(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_select_reduce(expr, ins, out_shape)

    _ensure_semantics("select_reduce", shape_rule, value_rule)

    assert (_is_sym_tensor(predicate) and _is_sym_tensor(on_true))
    inputs = [predicate, on_true]
    attrs: dict[str, Any] = {
        "reduce_cmd": reduce_cmd,
        "reduce_op": reduce_op,
        "reverse_pred": reverse_pred,
        "with_reduce": reduce_res is not None,
        "name": name,
    }
    if _is_sym_tensor(on_false):
        attrs["on_false_input_index"] = len(inputs)
        inputs.append(on_false)
    else:
        attrs["on_false_const"] = on_false
    return _new_sym_tensor("select_reduce", inputs, attrs, _default_out_shape(dst, on_true))


@semantics()
def tensor_copy(dst, src, engine=engine.unknown, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_same_as_first(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_copy(expr, ins, out_shape)

    _ensure_semantics("tensor_copy", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    return _new_sym_tensor("tensor_copy", [src], {"engine": engine, "name": name}, _default_out_shape(dst, src))


@semantics()
def tensor_copy_predicated(dst, src, predicate, reverse_pred=False, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        src_shape = ins[0]
        pred_shape = ins[1]
        ctx = _shape_ctx(*src_shape.dims)
        ctx.extend(_broadcast_shape(src_shape, pred_shape).ctx.facts)
        if len(ins) > 2:
            ctx.extend(_broadcast_shape(src_shape, ins[2]).ctx.facts)
        return ShapeResult(ShapeExpr(list(src_shape.dims)), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        src_sem, pred_sem = ins[0], ins[1]
        prior_dst = ins[2] if len(ins) > 2 else None
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = src_sem.ctx.merged(pred_sem.ctx, *(s.ctx for s in ins[2:]))
        pred = _call_broadcasted(pred_sem, out_shape, idx) != 0
        pred = z3.Not(pred) if expr.attrs.get("reverse_pred", False) else pred
        src_value = _call_broadcasted(src_sem, out_shape, idx)
        old_value = _call_broadcasted(prior_dst, out_shape, idx) if prior_dst is not None else z3.RealVal(0)
        ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(pred, src_value, old_value)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("tensor_copy_predicated", shape_rule, value_rule)

    assert (_is_sym_tensor(src) and _is_sym_tensor(predicate))
    inputs = [src, predicate]
    if _is_sym_tensor(dst):
        inputs.append(dst)
    return _new_sym_tensor(
        "tensor_copy_predicated",
        inputs,
        {"reverse_pred": reverse_pred, "name": name},
        _default_out_shape(dst, src),
    )


@semantics()
def tensor_partition_reduce(dst, op, data, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        if data_shape.rank == 0:
            return ShapeResult(ShapeExpr([]), Context([z3.BoolVal(False)]))
        out_dims = [z3.IntVal(1), *data_shape.dims[1:]] if data_shape.rank > 1 else [z3.IntVal(1)]
        return ShapeResult(ShapeExpr(out_dims), _shape_ctx(*data_shape.dims, *out_dims))

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        data_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = data_sem.ctx.merged()
        if data_sem.shape.rank == 1:
            j = z3.Int(f"{expr.name}_j")
            body_id = _fresh_body_id()
            n = data_sem.shape.dims[0]
            ctx.add(z3.ForAll([j], BODY1(z3.IntVal(body_id), z3.IntVal(0), j) == data_sem.fn(j)))
            ctx.add(z3.ForAll([j], out_fn(j) == REDUCE1(z3.IntVal(body_id), z3.IntVal(0), n)))
            return Semantics(expr.name, out_shape, out_fn, ctx, reduction=ReductionDesc(body_id, n, outer_rank=1))
        tail_idx = _index_vars(expr.name, out_shape.rank - 1, "k")
        p = z3.Int(f"{expr.name}_p")
        body_id = _fresh_body_id()
        partition_extent = data_sem.shape.dims[0]
        ctx.add(z3.ForAll([p, *tail_idx], BODY1(z3.IntVal(body_id), _linear_index(list(data_sem.shape.dims[1:]), tail_idx), p) == data_sem.fn(p, *tail_idx)))
        for_all_idx = [z3.Int(f"{expr.name}_o{k}") for k in range(out_shape.rank)]
        flat_tail = _linear_index(list(out_shape.dims[1:]), for_all_idx[1:]) if out_shape.rank > 1 else z3.IntVal(0)
        ctx.add(z3.ForAll(for_all_idx, z3.Implies(for_all_idx[0] == 0, out_fn(*for_all_idx) == REDUCE1(z3.IntVal(body_id), flat_tail, partition_extent))))
        return Semantics(
            expr.name,
            out_shape,
            out_fn,
            ctx,
            reduction=ReductionDesc(body_id, partition_extent, outer_rank=builtins.max(1, out_shape.rank - 1)),
        )

    _ensure_semantics("tensor_partition_reduce", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    out_shape = _default_out_shape(dst, data)
    if out_shape == data.shape:
        out_shape = (z3.IntVal(1), *data.shape[1:]) if data.rank > 1 else (z3.IntVal(1),)
    return _new_sym_tensor("tensor_partition_reduce", [data], {"op": op, "name": name}, out_shape)


@semantics()
def tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_tensor_reduce(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_tensor_reduce(expr, ins, out_shape)

    _ensure_semantics("tensor_reduce", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    return _new_sym_tensor(
        "tensor_reduce",
        [data],
        {"op": op, "axis": axis, "negate": negate, "keepdims": keepdims, "name": name},
        _default_out_shape(dst, data),
    )


@semantics()
def tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, engine=engine.unknown, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_tensor_scalar(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_tensor_scalar(expr, ins, out_shape)

    _ensure_semantics("tensor_scalar", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {
        "op0": op0,
        "reverse0": reverse0,
        "op1": op1,
        "reverse1": reverse1,
        "engine": engine,
        "name": name,
    }
    if _is_sym_tensor(operand0):
        attrs["operand0_input_index"] = len(inputs)
        inputs.append(operand0)
    else:
        attrs["operand0_const"] = operand0
    if _is_sym_tensor(operand1):
        attrs["operand1_input_index"] = len(inputs)
        inputs.append(operand1)
    elif operand1 is not None:
        attrs["operand1_const"] = operand1
    return _new_sym_tensor("tensor_scalar", inputs, attrs, _default_out_shape(dst, data))


@semantics()
def tensor_scalar_cumulative(dst, src, op0, op1, imm0, imm1=None, reduce_cmd=reduce_cmd.reset_reduce, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        if len(ins) > 1:
            ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
        if len(ins) > 2:
            ctx.extend(_broadcast_shape(data_shape, ins[2]).ctx.facts)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        src_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = src_sem.ctx.merged(*[s.ctx for s in ins[1:]])
        if out_shape.rank != 2:
            ctx.add(z3.BoolVal(False))
            return Semantics(expr.name, out_shape, out_fn, ctx)
        i = z3.Int(f"{expr.name}_i")
        j = z3.Int(f"{expr.name}_j")
        prev = z3.Function(f"SCAN_{expr.name}", z3.IntSort(), z3.IntSort(), z3.RealSort())
        idx = [i, j]
        imm0_value = _operand_value(expr.attrs, "imm0_const", "imm0_input_index", ins, out_shape, idx)
        data_value = src_sem.fn(i, j)
        lhs0, rhs0 = (imm0_value, data_value) if expr.attrs.get("reverse0", False) else (data_value, imm0_value)
        step0 = _apply_binary(expr.attrs.get("op0"), lhs0, rhs0)
        init_value = _operand_value(expr.attrs, "imm1_const", "imm1_input_index", ins, ShapeExpr([out_shape.dims[0], z3.IntVal(1)]), [i, z3.IntVal(0)])
        seed = z3.If(
            expr.attrs.get("reduce_cmd") == reduce_cmd.load_reduce,
            init_value,
            z3.If(
                _operand_to_expr(expr.attrs.get("op1")) in ("multiply", "mul"),
                z3.RealVal(1),
                z3.If(_operand_to_expr(expr.attrs.get("op1")) in ("minimum", "min"), z3.RealVal(_LARGE_POSITIVE_SENTINEL), z3.RealVal(0)),
            ),
        )
        first = _apply_binary(expr.attrs.get("op1"), step0, seed) if not expr.attrs.get("reverse1", False) else _apply_binary(expr.attrs.get("op1"), seed, step0)
        ctx.add(z3.ForAll([i], prev(i, z3.IntVal(0)) == first))
        stepj = _apply_binary(expr.attrs.get("op1"), step0, prev(i, j - 1)) if not expr.attrs.get("reverse1", False) else _apply_binary(expr.attrs.get("op1"), prev(i, j - 1), step0)
        ctx.add(z3.ForAll([i, j], z3.Implies(j > 0, prev(i, j) == stepj)))
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == prev(i, j)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("tensor_scalar_cumulative", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    inputs = [src]
    attrs: dict[str, Any] = {"op0": op0, "op1": op1, "reverse0": False, "reverse1": False, "reduce_cmd": reduce_cmd, "name": name}
    if _is_sym_tensor(imm0):
        attrs["imm0_input_index"] = len(inputs)
        inputs.append(imm0)
    else:
        attrs["imm0_const"] = imm0
    if _is_sym_tensor(imm1):
        attrs["imm1_input_index"] = len(inputs)
        inputs.append(imm1)
    elif imm1 is not None:
        attrs["imm1_const"] = imm1
    return _new_sym_tensor("tensor_scalar_cumulative", inputs, attrs, _default_out_shape(dst, src))


@semantics()
def tensor_scalar_reduce(dst, data, op0, operand0, reduce_op, reduce_res, reverse0=False, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        if len(ins) > 1:
            ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        data_sem = ins[0]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        idx = _index_vars(expr.name, out_shape.rank)
        ctx = data_sem.ctx.merged(*[s.ctx for s in ins[1:]])
        data_value = _call_broadcasted(data_sem, out_shape, idx)
        operand_value = _operand_value(expr.attrs, "operand0_const", "operand0_input_index", ins, out_shape, idx)
        lhs, rhs = (operand_value, data_value) if expr.attrs.get("reverse0", False) else (data_value, operand_value)
        tmp = _apply_binary(expr.attrs.get("op0"), lhs, rhs)
        ctx.add(z3.ForAll(idx, out_fn(*idx) == tmp))
        reduction = None
        if out_shape.rank == 2:
            body_id = _fresh_body_id()
            i = z3.Int(f"{expr.name}_ri")
            k = z3.Int(f"{expr.name}_rk")
            n = out_shape.dims[1]
            ctx.add(z3.ForAll([i, k], BODY1(z3.IntVal(body_id), i, k) == out_fn(i, k)))
            reduction = ReductionDesc(body_id, n, outer_rank=1)
        return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)

    _ensure_semantics("tensor_scalar_reduce", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    inputs = [data]
    attrs: dict[str, Any] = {"op0": op0, "reduce_op": reduce_op, "reverse0": reverse0, "name": name}
    if _is_sym_tensor(operand0):
        attrs["operand0_input_index"] = len(inputs)
        inputs.append(operand0)
    else:
        attrs["operand0_const"] = operand0
    return _new_sym_tensor("tensor_scalar_reduce", inputs, attrs, _default_out_shape(dst, data))


@semantics()
def tensor_tensor(dst, data1, data2, op, engine=engine.unknown, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_tensor_tensor(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_tensor_tensor(expr, ins, out_shape)

    _ensure_semantics("tensor_tensor", shape_rule, value_rule)

    assert (_is_sym_tensor(data1) and _is_sym_tensor(data2))
    return _new_sym_tensor(
        "tensor_tensor",
        [data1, data2],
        {"op": op, "engine": engine, "name": name},
        _default_out_shape(dst, data1),
    )


@semantics()
def tensor_tensor_scan(dst, data0, data1, initial, op0, op1, reverse0=False, reverse1=False, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        data_shape = ins[0]
        ctx = _shape_ctx(*data_shape.dims)
        ctx.extend(_broadcast_shape(data_shape, ins[1]).ctx.facts)
        if len(ins) > 2:
            ctx.extend(_broadcast_shape(ShapeExpr([data_shape.dims[0], z3.IntVal(1)]), ins[2]).ctx.facts)
        return ShapeResult(ShapeExpr(list(data_shape.dims)), ctx)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        a, b = ins[0], ins[1]
        out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
        ctx = a.ctx.merged(b.ctx, *(s.ctx for s in ins[2:]))
        if out_shape.rank != 2:
            ctx.add(z3.BoolVal(False))
            return Semantics(expr.name, out_shape, out_fn, ctx)
        i = z3.Int(f"{expr.name}_i")
        j = z3.Int(f"{expr.name}_j")
        prev = z3.Function(f"SCAN_{expr.name}", z3.IntSort(), z3.IntSort(), z3.RealSort())
        init = _operand_value(expr.attrs, "initial_const", "initial_input_index", ins, ShapeExpr([out_shape.dims[0], z3.IntVal(1)]), [i, z3.IntVal(0)])
        first_a = a.fn(i, z3.IntVal(0))
        lhs0_first, rhs0_first = (init, first_a) if expr.attrs.get("reverse0", False) else (first_a, init)
        first_tmp = _apply_binary(expr.attrs.get("op0"), lhs0_first, rhs0_first)
        first_b = b.fn(i, z3.IntVal(0))
        lhs1_first, rhs1_first = (first_b, first_tmp) if expr.attrs.get("reverse1", False) else (first_tmp, first_b)
        first_step = _apply_binary(expr.attrs.get("op1"), lhs1_first, rhs1_first)
        prev_or_init = prev(i, j - 1)
        lhs0, rhs0 = (prev_or_init, a.fn(i, j)) if expr.attrs.get("reverse0", False) else (a.fn(i, j), prev_or_init)
        tmp = _apply_binary(expr.attrs.get("op0"), lhs0, rhs0)
        lhs1, rhs1 = (b.fn(i, j), tmp) if expr.attrs.get("reverse1", False) else (tmp, b.fn(i, j))
        step = _apply_binary(expr.attrs.get("op1"), lhs1, rhs1)
        ctx.add(z3.ForAll([i], prev(i, z3.IntVal(0)) == first_step))
        ctx.add(z3.ForAll([i, j], z3.Implies(j > 0, prev(i, j) == step)))
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == prev(i, j)))
        return Semantics(expr.name, out_shape, out_fn, ctx)

    _ensure_semantics("tensor_tensor_scan", shape_rule, value_rule)

    assert (_is_sym_tensor(data0) and _is_sym_tensor(data1))
    inputs = [data0, data1]
    attrs: dict[str, Any] = {"op0": op0, "op1": op1, "reverse0": reverse0, "reverse1": reverse1, "name": name}
    if _is_sym_tensor(initial):
        attrs["initial_input_index"] = len(inputs)
        inputs.append(initial)
    else:
        attrs["initial_const"] = initial
    return _new_sym_tensor("tensor_tensor_scan", inputs, attrs, _default_out_shape(dst, data0))


def _call_broadcasted(sem: Semantics, out_shape: ShapeExpr, indices: list[z3.ArithRef]) -> z3.ArithRef:
    mapped = _broadcast_indices(sem, out_shape, indices)
    return sem.fn(*mapped) if mapped else sem.fn(z3.IntVal(0))


def _shape_same_as_first(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    return ShapeResult(ShapeExpr(list(a.dims)), Context([d > 0 for d in a.dims]))


def _compile_copy(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    src = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = src.ctx.merged()
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == _call_broadcasted(src, out_shape, idx)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_tensor_tensor(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    return _broadcast_shape(ins[0], ins[1])


def _compile_tensor_tensor(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    a, b = ins
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged(b.ctx)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    av = _call_broadcasted(a, out_shape, idx)
    bv = _call_broadcasted(b, out_shape, idx)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == _apply_binary(expr.attrs.get("op"), av, bv)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_tensor_scalar(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    data = ins[0]
    ctx = Context([d > 0 for d in data.dims])
    if len(ins) > 1:
        b0 = _broadcast_shape(data, ins[1])
        ctx.extend(b0.ctx.facts)
    if len(ins) > 2:
        b1 = _broadcast_shape(data, ins[2])
        ctx.extend(b1.ctx.facts)
    return ShapeResult(ShapeExpr(list(data.dims)), ctx)


def _operand_value(
    attrs: dict[str, Any],
    constant_attr_key: str,
    input_index_attr_key: str,
    ins: list[Semantics],
    out_shape: ShapeExpr,
    indices: list[z3.ArithRef],
) -> z3.ArithRef:
    if input_index_attr_key in attrs:
        return _call_broadcasted(ins[int(attrs[input_index_attr_key])], out_shape, indices)
    return _as_scalar(attrs.get(constant_attr_key))


def _compile_tensor_scalar(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    data = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = data.ctx.merged(*[s.ctx for s in ins[1:]])
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    dv = _call_broadcasted(data, out_shape, idx)
    op0 = _operand_value(expr.attrs, "operand0_const", "operand0_input_index", ins, out_shape, idx)
    lhs0, rhs0 = (op0, dv) if expr.attrs.get("reverse0", False) else (dv, op0)
    tmp = _apply_binary(expr.attrs.get("op0"), lhs0, rhs0)
    if expr.attrs.get("op1") is not None:
        op1 = _operand_value(expr.attrs, "operand1_const", "operand1_input_index", ins, out_shape, idx)
        lhs1, rhs1 = (op1, tmp) if expr.attrs.get("reverse1", False) else (tmp, op1)
        tmp = _apply_binary(expr.attrs.get("op1"), lhs1, rhs1)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == tmp))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_nc_transpose(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    if a.rank < 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    return ShapeResult(ShapeExpr([a.dims[1], a.dims[0], *a.dims[2:]]), ctx)


def _compile_nc_transpose(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged()
    reduction = None
    if out_shape.rank < 2:
        ctx.add(z3.BoolVal(False))
        return Semantics(expr.name, out_shape, out_fn, ctx)
    i = z3.Int(f"{expr.name}_i")
    j = z3.Int(f"{expr.name}_j")
    if out_shape.rank == 2:
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == a.fn(j, i)))
    else:
        rest = [z3.Int(f"{expr.name}_k{t}") for t in range(out_shape.rank - 2)]
        out_idx = [i, j, *rest]
        src_idx = [j, i, *rest]
        ctx.add(z3.ForAll(out_idx, out_fn(*out_idx) == a.fn(*src_idx)))
    if out_shape.rank == 2 and a.reduction is not None and a.reduction.outer_rank == 2:
        body_id = _fresh_body_id()
        k = z3.Int(f"{expr.name}_rk")
        source_body = z3.IntVal(a.reduction.body_id)
        target_body = z3.IntVal(body_id)
        ctx.add(z3.ForAll([i, j, k], BODY2(target_body, i, j, k) == BODY2(source_body, j, i, k)))
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == REDUCE2(target_body, i, j, a.reduction.extent)))
        reduction = ReductionDesc(body_id, a.reduction.extent, outer_rank=2)
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _shape_tensor_reduce(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    if a.rank != 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    axis_attr = attrs.get("axis", 1)
    axis = axis_attr[0] if isinstance(axis_attr, (list, tuple)) else axis_attr
    keep = bool(attrs.get("keepdims", False))
    if axis == 1:
        return ShapeResult(ShapeExpr([a.dims[0], z3.IntVal(1)] if keep else [a.dims[0]]), ctx)
    if axis == 0:
        return ShapeResult(ShapeExpr([z3.IntVal(1), a.dims[1]] if keep else [a.dims[1]]), ctx)
    ctx.add(z3.BoolVal(False))
    return ShapeResult(ShapeExpr(list(a.dims)), ctx)


def _compile_tensor_reduce(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    axis_attr = expr.attrs.get("axis", 1)
    axis = axis_attr[0] if isinstance(axis_attr, (list, tuple)) else axis_attr
    keep = bool(expr.attrs.get("keepdims", False))
    negate = bool(expr.attrs.get("negate", False))
    body_id = _fresh_body_id()
    ctx = a.ctx.merged()
    i = z3.Int(f"{expr.name}_i")
    j = z3.Int(f"{expr.name}_j")
    k = z3.Int(f"{expr.name}_k")
    if axis == 1:
        n = a.shape.dims[1]
        ctx.add(z3.ForAll([i, k], BODY1(z3.IntVal(body_id), i, k) == a.fn(i, k)))
        val = REDUCE1(z3.IntVal(body_id), i, n)
        val = z3.If(negate, -val, val)
        if keep:
            ctx.add(z3.ForAll([i, j], z3.Implies(j == 0, out_fn(i, j) == val)))
        else:
            ctx.add(z3.ForAll([i], out_fn(i) == val))
        return Semantics(expr.name, out_shape, out_fn, ctx, reduction=ReductionDesc(body_id, n, outer_rank=1))
    if axis == 0:
        m = a.shape.dims[0]
        ctx.add(z3.ForAll([j, k], BODY1(z3.IntVal(body_id), j, k) == a.fn(k, j)))
        val = REDUCE1(z3.IntVal(body_id), j, m)
        val = z3.If(negate, -val, val)
        if keep:
            ctx.add(z3.ForAll([i, j], z3.Implies(i == 0, out_fn(i, j) == val)))
        else:
            ctx.add(z3.ForAll([j], out_fn(j) == val))
        return Semantics(expr.name, out_shape, out_fn, ctx, reduction=ReductionDesc(body_id, m, outer_rank=1))
    ctx.add(z3.BoolVal(False))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_unary_same(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    return _shape_same_as_first(ins, attrs)


def _compile_reciprocal(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx = a.ctx.merged()
    x = _call_broadcasted(a, out_shape, idx)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(x == 0, z3.RealVal(0), z3.RealVal(1) / x)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _compile_activation(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    data = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx = data.ctx.merged(*[s.ctx for s in ins[1:]])
    scale = _operand_value(expr.attrs, "scale", "scale_input_index", ins, out_shape, idx)
    bias = _operand_value(expr.attrs, "bias_const", "bias_input_index", ins, out_shape, idx)
    pre = _call_broadcasted(data, out_shape, idx) * scale + bias
    act, extra = _apply_activation(expr.attrs.get("op"), pre)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == act))
    for fact in extra:
        ctx.add(z3.ForAll(idx, fact))
    reduction = None
    if expr.attrs.get("with_reduce", False) and out_shape.rank == 2:
        body_id = _fresh_body_id()
        i = z3.Int(f"{expr.name}_ri")
        k = z3.Int(f"{expr.name}_rk")
        n = out_shape.dims[1]
        ctx.add(z3.ForAll([i, k], BODY1(z3.IntVal(body_id), i, k) == out_fn(i, k)))
        reduction = ReductionDesc(body_id, n, outer_rank=1)
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _compile_activation_reduce(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    return _compile_activation(expr, ins, out_shape)


def _shape_activation(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    data = ins[0]
    ctx = Context([d > 0 for d in data.dims])
    if len(ins) > 1:
        ctx.extend(_broadcast_shape(data, ins[1]).ctx.facts)
    if len(ins) > 2:
        ctx.extend(_broadcast_shape(data, ins[2]).ctx.facts)
    return ShapeResult(ShapeExpr(list(data.dims)), ctx)


def _compile_exponential(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    src = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx = src.ctx.merged(*[s.ctx for s in ins[1:]])
    maxv = _operand_value(expr.attrs, "max_value", "max_input_index", ins, out_shape, idx)
    pre = _call_broadcasted(src, out_shape, idx) - maxv
    act, extra = _apply_activation("exp", pre)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == act))
    for fact in extra:
        ctx.add(z3.ForAll(idx, fact))
    reduction = None
    if expr.attrs.get("with_reduce", False) and out_shape.rank == 2:
        body_id = _fresh_body_id()
        i = z3.Int(f"{expr.name}_ri")
        k = z3.Int(f"{expr.name}_rk")
        n = out_shape.dims[1]
        ctx.add(z3.ForAll([i, k], BODY1(z3.IntVal(body_id), i, k) == out_fn(i, k)))
        reduction = ReductionDesc(body_id, n, outer_rank=1)
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _compare_bool(op: Any, lhs: z3.ArithRef, rhs: z3.ArithRef) -> z3.BoolRef:
    opn = _operand_to_expr(op)
    if opn == "equal":
        return lhs == rhs
    if opn == "less":
        return lhs < rhs
    if opn == "less_equal":
        return lhs <= rhs
    if opn == "greater":
        return lhs > rhs
    if opn == "greater_equal":
        return lhs >= rhs
    key = str(opn)
    with _UF_LOCK:
        if key not in _COMPARE_UFS:
            _COMPARE_UFS[key] = z3.Function(f"NKI_CMP_{len(_COMPARE_UFS)}", z3.RealSort(), z3.RealSort(), z3.BoolSort())
        fn = _COMPARE_UFS[key]
    return fn(lhs, rhs)


def _compile_select_reduce(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    pred, on_true = ins[0], ins[1]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    ctx = pred.ctx.merged(on_true.ctx, *[s.ctx for s in ins[2:]])
    p = _call_broadcasted(pred, out_shape, idx) != 0
    p = z3.Not(p) if expr.attrs.get("reverse_pred", False) else p
    t = _call_broadcasted(on_true, out_shape, idx)
    f = _operand_value(expr.attrs, "on_false_const", "on_false_input_index", ins, out_shape, idx)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(p, t, f)))
    reduction = None
    if expr.attrs.get("with_reduce", False) and out_shape.rank == 2:
        body_id = _fresh_body_id()
        i = z3.Int(f"{expr.name}_ri")
        k = z3.Int(f"{expr.name}_rk")
        n = out_shape.dims[1]
        ctx.add(z3.ForAll([i, k], BODY1(z3.IntVal(body_id), i, k) == out_fn(i, k)))
        reduction = ReductionDesc(body_id, n, outer_rank=1)
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=reduction)


def _extract_iota_step(pattern: Any) -> int:
    """Extract the innermost iota step, defaulting to 1 when the pattern omits it."""
    if isinstance(pattern, (list, tuple)) and pattern:
        tail_pos = len(pattern) - 1
        tail = pattern[tail_pos]
        if isinstance(tail, (list, tuple)) and len(tail) >= 1:
            try:
                return int(tail[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"iota pattern step values must be integers, got {tail[0]!r} in entry {tail_pos} of {pattern!r}"
                ) from exc
    return 1


def _shape_public_attr_or_first(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    out_shape = attrs.get("out_shape")
    if out_shape is not None:
        return _shape_from_out(tuple(out_shape))
    shape = attrs.get("shape")
    if shape is not None:
        return _shape_from_out(tuple(shape))
    if ins:
        return _shape_same_as_first(ins, attrs)
    return _shape_from_out((z3.IntVal(1),))


def _compile_public_opaque(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = Context()
    for sem in ins:
        ctx = ctx.merged(sem.ctx)
    opaque = z3.Function(
        f"PUBLIC_{expr.name}",
        *([z3.RealSort()] * len(ins)),
        *([z3.IntSort()] * builtins.max(1, out_shape.rank)),
        z3.RealSort(),
    )
    input_values = [_call_broadcasted(sem, out_shape, idx) for sem in ins]
    index_args = idx if idx else [z3.IntVal(0)]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == opaque(*(input_values + index_args))))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_unary(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    return _shape_same_as_first(ins, attrs)


def _compile_public_unary(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = a.ctx.merged()
    value, facts = _apply_activation(expr.attrs.get("op", expr.op), _call_broadcasted(a, out_shape, idx))
    ctx.extend(facts)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == value))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_binary(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    if len(ins) >= 2:
        return _shape_tensor_tensor(ins[:2], attrs)
    if len(ins) == 1:
        return _shape_same_as_first(ins, attrs)
    return _shape_from_out(attrs.get("out_shape", (z3.IntVal(1),)))


def _compile_public_binary(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = Context()
    for sem in ins:
        ctx = ctx.merged(sem.ctx)
    if len(ins) >= 2:
        lhs = _call_broadcasted(ins[0], out_shape, idx)
        rhs = _call_broadcasted(ins[1], out_shape, idx)
    elif len(ins) == 1:
        lhs = _call_broadcasted(ins[0], out_shape, idx)
        rhs = _as_scalar(
            expr.attrs.get(
                "scalar",
                expr.attrs.get(
                    "rhs",
                    expr.attrs.get("value", expr.attrs.get("operand0_const", 0)),
                ),
            )
        )
        if expr.attrs.get("reverse", False):
            lhs, rhs = rhs, lhs
    else:
        return _compile_public_opaque(expr, ins, out_shape)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == _apply_binary(expr.attrs.get("op", expr.op), lhs, rhs)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_reduce(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    axis_attr = attrs.get("axis")
    if axis_attr is None:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    axes = [axis_attr] if isinstance(axis_attr, int) else list(axis_attr)
    norm_axes: list[int] = []
    for axis in axes:
        norm_axis = axis + a.rank if axis < 0 else axis
        if norm_axis < 0 or norm_axis >= a.rank:
            ctx.add(z3.BoolVal(False))
            return ShapeResult(ShapeExpr(list(a.dims)), ctx)
        norm_axes.append(norm_axis)
    keepdims = bool(attrs.get("keepdims", False))
    out_dims: list[z3.ArithRef] = []
    for dim_index, dim in enumerate(a.dims):
        if dim_index in norm_axes:
            if keepdims:
                out_dims.append(z3.IntVal(1))
        else:
            out_dims.append(dim)
    return ShapeResult(ShapeExpr(out_dims if out_dims else [z3.IntVal(1)]), ctx)


def _compile_public_reduce(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    a = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = a.ctx.merged()
    opaque = z3.Function(
        f"PUBLIC_REDUCE_{expr.name}",
        *([z3.IntSort()] * builtins.max(1, out_shape.rank)),
        z3.RealSort(),
    )
    index_args = idx if idx else [z3.IntVal(0)]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == opaque(*index_args)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_broadcast_to(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    target_shape = attrs.get("shape") or attrs.get("out_shape")
    if target_shape is None:
        return _shape_same_as_first(ins, attrs)
    out = _shape_from_out(tuple(target_shape))
    if not ins:
        return out
    compat = _broadcast_shape(ins[0], out.out)
    return ShapeResult(out.out, compat.ctx)


def _shape_public_expand_dims(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    axis = int(attrs.get("axis", 0))
    rank = a.rank + 1
    norm_axis = axis + rank if axis < 0 else axis
    if norm_axis < 0 or norm_axis > a.rank:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    dims = list(a.dims)
    dims.insert(norm_axis, z3.IntVal(1))
    return ShapeResult(ShapeExpr(dims), ctx)


def _compile_public_expand_dims(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    src = ins[0]
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = src.ctx.merged()
    axis = int(expr.attrs.get("axis", 0))
    norm_axis = axis + out_shape.rank if axis < 0 else axis
    src_idx = [dim for i, dim in enumerate(idx) if i != norm_axis]
    ctx.add(z3.ForAll(idx, out_fn(*idx) == src.fn(*src_idx)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_transpose2d(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a = ins[0]
    ctx = Context([d > 0 for d in a.dims])
    if a.rank != 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    return ShapeResult(ShapeExpr([a.dims[1], a.dims[0]]), ctx)


def _shape_public_where(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    if len(ins) != 3:
        return ShapeResult(ShapeExpr([z3.IntVal(1)]), Context([z3.BoolVal(False)]))
    xy = _broadcast_shape(ins[1], ins[2])
    cxy = _broadcast_shape(ins[0], xy.out)
    return ShapeResult(cxy.out, xy.ctx.merged(cxy.ctx))


def _compile_public_where(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    cond_sem, x_sem, y_sem = ins
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    idx = _index_vars(expr.name, out_shape.rank)
    ctx = cond_sem.ctx.merged(x_sem.ctx, y_sem.ctx)
    cond_value = _call_broadcasted(cond_sem, out_shape, idx) != 0
    x_value = _call_broadcasted(x_sem, out_shape, idx)
    y_value = _call_broadcasted(y_sem, out_shape, idx)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == z3.If(cond_value, x_value, y_value)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_public_matmul(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    a, b = ins
    ctx = Context([d > 0 for d in a.dims + b.dims])
    if a.rank != 2 or b.rank != 2:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    if bool(attrs.get("transpose_x", False)):
        ctx.add(a.dims[0] == b.dims[0])
        return ShapeResult(ShapeExpr([a.dims[1], b.dims[1]]), ctx)
    ctx.add(a.dims[1] == b.dims[0])
    return ShapeResult(ShapeExpr([a.dims[0], b.dims[1]]), ctx)


def _compile_public_matmul(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    a, b = ins
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged(b.ctx)
    if a.shape.rank != 2 or b.shape.rank != 2 or out_shape.rank != 2:
        ctx.add(z3.BoolVal(False))
        return Semantics(expr.name, out_shape, out_fn, ctx)
    m = z3.Int(f"{expr.name}_m")
    n = z3.Int(f"{expr.name}_n")
    k = z3.Int(f"{expr.name}_k")
    body_id = _fresh_body_id()
    if bool(expr.attrs.get("transpose_x", False)):
        ctx.add(z3.ForAll([m, n, k], BODY2(z3.IntVal(body_id), m, n, k) == a.fn(k, m) * b.fn(k, n)))
        extent = a.shape.dims[0]
    else:
        ctx.add(z3.ForAll([m, n, k], BODY2(z3.IntVal(body_id), m, n, k) == a.fn(m, k) * b.fn(k, n)))
        extent = a.shape.dims[1]
    ctx.add(z3.ForAll([m, n], out_fn(m, n) == REDUCE2(z3.IntVal(body_id), m, n, extent)))
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=ReductionDesc(body_id, extent, outer_rank=2))


def _shape_public_gather_flattened(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    if len(ins) != 2:
        return ShapeResult(ShapeExpr([z3.IntVal(1)]), Context([z3.BoolVal(False)]))
    data_shape, idx_shape = ins
    dims = [data_shape.dims[0], *idx_shape.dims[1:]] if idx_shape.rank >= 1 else list(data_shape.dims)
    return ShapeResult(ShapeExpr(dims), _shape_ctx(*data_shape.dims, *idx_shape.dims))


def _public_broadcast_shape_tuple(a_shape: tuple[Any, ...], b_shape: tuple[Any, ...]) -> tuple[Any, ...]:
    rank = builtins.max(len(a_shape), len(b_shape))
    a_dims = [1] * (rank - len(a_shape)) + list(a_shape)
    b_dims = [1] * (rank - len(b_shape)) + list(b_shape)
    out: list[Any] = []
    for a_dim, b_dim in zip(a_dims, b_dims):
        if isinstance(a_dim, int) and isinstance(b_dim, int):
            if a_dim == 1:
                out.append(b_dim)
                continue
            if b_dim == 1 or a_dim == b_dim:
                out.append(a_dim)
                continue
        out.append(z3.If(_to_dim(a_dim) == z3.IntVal(1), _to_dim(b_dim), _to_dim(a_dim)))
    return tuple(out)


def _public_reduce_out_shape(shape: tuple[Any, ...], axis: Any, keepdims: bool) -> tuple[Any, ...]:
    if axis is None:
        return shape
    rank = len(shape)
    axes = [axis] if isinstance(axis, int) else list(axis)
    norm_axes: set[int] = set()
    for ax in axes:
        if isinstance(ax, int):
            norm_axes.add(ax + rank if ax < 0 else ax)
    out: list[Any] = []
    for i, dim in enumerate(shape):
        if i in norm_axes:
            if keepdims:
                out.append(1)
        else:
            out.append(dim)
    return tuple(out) if out else (1,)


def _public_expand_dims_out_shape(shape: tuple[Any, ...], axis: int) -> tuple[Any, ...]:
    dims = list(shape)
    rank = len(dims) + 1
    norm_axis = axis + rank if axis < 0 else axis
    if norm_axis < 0 or norm_axis > len(dims):
        return tuple(dims)
    dims.insert(norm_axis, 1)
    return tuple(dims)


def loop_reduce(data: Any, op: Any, loop_indices: list[Any], mask=None, dtype=None, name=None):
    if not _is_sym_tensor(data):
        return data
    return _new_sym_tensor("loop_reduce", [data], {"op": op, "name": name}, data.shape)


def _register_public_semantics() -> None:
    unary_ops = {
        "abs",
        "arctan",
        "ceil",
        "copy",
        "cos",
        "erf",
        "erf_dx",
        "exp",
        "floor",
        "gelu",
        "gelu_apprx_sigmoid",
        "gelu_apprx_sigmoid_dx",
        "gelu_apprx_tanh",
        "gelu_dx",
        "invert",
        "log",
        "logical_not",
        "mish",
        "negative",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "sign",
        "silu",
        "silu_dx",
        "sin",
        "softmax",
        "softplus",
        "sqrt",
        "square",
        "tan",
        "tanh",
        "trunc",
    }
    for op_name in unary_ops:
        compile_rule = _compile_reciprocal if op_name == "reciprocal" else _compile_public_unary
        _ensure_semantics(op_name, _shape_public_unary, compile_rule)

    binary_ops = {
        "add",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "divide",
        "equal",
        "fmod",
        "greater",
        "greater_equal",
        "left_shift",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "mod",
        "multiply",
        "not_equal",
        "power",
        "right_shift",
        "subtract",
    }
    for op_name in binary_ops:
        _ensure_semantics(op_name, _shape_public_binary, _compile_public_binary)

    for op_name in {"all", "max", "mean", "min", "prod", "sum", "var"}:
        _ensure_semantics(op_name, _shape_public_reduce, _compile_public_reduce)

    for op_name in {
        "dropout",
        "ds",
        "gather_flattened",
        "rms_norm",
        "loop_reduce",
        "store",
    }:
        shape_rule = _shape_public_gather_flattened if op_name == "gather_flattened" else _shape_public_attr_or_first
        compile_rule = _compile_copy if op_name == "store" else _compile_public_opaque
        _ensure_semantics(op_name, shape_rule, compile_rule)

    _ensure_semantics("broadcast_to", _shape_public_broadcast_to, _compile_copy)
    _ensure_semantics("expand_dims", _shape_public_expand_dims, _compile_public_expand_dims)
    _ensure_semantics("load", _shape_same_as_first, _compile_copy)
    _ensure_semantics("load_transpose2d", _shape_public_transpose2d, _compile_nc_transpose)
    _ensure_semantics("matmul", _shape_public_matmul, _compile_public_matmul)
    _ensure_semantics("transpose", _shape_public_transpose2d, _compile_nc_transpose)
    _ensure_semantics("where", _shape_public_where, _compile_public_where)


_register_public_semantics()






class AxonArray:
    def __init__(self, node_id: str, shape: tuple[int, ...]):
        self.node_id = node_id
        self.shape = shape

    def _binary(self, other: Any, op: str) -> "AxonArray":
        if op == "matmul":
            if not isinstance(other, AxonArray):
                raise TypeError("matmul expects AxonArray operand")
            if len(self.shape) != 2 or len(other.shape) != 2:
                raise ValueError("matmul expects rank-2 inputs")
            return AxonArray(_gen_id(op), (self.shape[0], other.shape[1]))

        if isinstance(other, AxonArray):
            out_shape = tuple(_to_int_dim(d) for d in _public_broadcast_shape_tuple(self.shape, other.shape))
        else:
            out_shape = self.shape
        return AxonArray(_gen_id(op), out_shape)

    def __add__(self, other: Any) -> "AxonArray":
        return self._binary(other, "add")

    def __mul__(self, other: Any) -> "AxonArray":
        return self._binary(other, "mul")

    def __truediv__(self, other: Any) -> "AxonArray":
        return self._binary(other, "div")

    def __matmul__(self, other: Any) -> "AxonArray":
        return self._binary(other, "matmul")

    def sum(self, axis: Any = None, keep_dims: bool = False) -> "AxonArray":
        out_shape = _public_reduce_out_shape(self.shape, axis, keep_dims)
        return AxonArray(_gen_id("reduce_sum"), tuple(_to_int_dim(d) for d in out_shape))

    def broadcast_like(self, other: "AxonArray") -> "AxonArray":
        return AxonArray(_gen_id("broadcast"), other.shape)

    def sqrt(self) -> "AxonArray":
        return AxonArray(_gen_id("sqrt"), self.shape)

    def exp(self) -> "AxonArray":
        return AxonArray(_gen_id("exp"), self.shape)

    def transpose(self) -> "AxonArray":
        if len(self.shape) == 2:
            return AxonArray(_gen_id("transpose"), (self.shape[1], self.shape[0]))
        return AxonArray(_gen_id("transpose"), self.shape)

    def relu(self) -> "AxonArray":
        return AxonArray(_gen_id("relu"), self.shape)

    def silu(self) -> "AxonArray":
        return AxonArray(_gen_id("silu"), self.shape)


@dataclass(eq=True, frozen=True)
class NodeSig:
    id: str
    op: str
    inputs: tuple[str, ...]
    attrs: tuple[tuple[str, Any], ...]


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    shape: Optional[tuple[int, ...]] = None

    def sig(self) -> NodeSig:
        return NodeSig(self.id, self.op, tuple(self.inputs), tuple(sorted(self.attrs.items())))


@dataclass
class nuGraph:
    nodes: list[Node]

    def position(self, node: Node) -> int:
        for i, n in enumerate(self.nodes):
            if n.id == node.id:
                return i
        raise ValueError("node not found")

    def node_at(self, pos: int) -> Node:
        return self.nodes[pos]

    def successors(self, node: Node) -> list[Node]:
        return [n for n in self.nodes if node.id in n.inputs]

    def clone(self) -> "nuGraph":
        return nuGraph([Node(n.id, n.op, list(n.inputs), dict(n.attrs), n.shape) for n in self.nodes])

    def __hash__(self):
        return hash(tuple(n.sig() for n in self.nodes))

    def __eq__(self, other):
        return isinstance(other, nuGraph) and tuple(n.sig() for n in self.nodes) == tuple(n.sig() for n in other.nodes)


def graph_signature(G: nuGraph) -> str:
    return " | ".join([f"{n.id}:{n.op}({','.join(n.inputs)})" for n in G.nodes])


def _to_int_dim(d: Any) -> Any:
    if isinstance(d, int):
        return d
    if isinstance(d, z3.IntNumRef):
        return d.as_long()
    return d


def _dims_equal(a: Any, b: Any) -> bool:
    lhs = _to_dim(a) if isinstance(a, int) else a
    rhs = _to_dim(b) if isinstance(b, int) else b
    return _arith_equal(lhs, rhs)


def annotate_shapes_concrete(G: nuGraph) -> nuGraph:
    shapes: dict[str, tuple[Any, ...]] = {}
    for n in G.nodes:
        if n.op == "input":
            shape = tuple(_to_int_dim(d) for d in n.attrs.get("shape", n.shape or tuple()))
        elif n.op == "reduce_sum" and n.inputs:
            in_shape = shapes.get(n.inputs[0], tuple())
            keep_dims = bool(n.attrs.get("keep_dims", n.attrs.get("keepdims", False)))
            shape = tuple(_to_int_dim(d) for d in _public_reduce_out_shape(in_shape, n.attrs.get("axis"), keep_dims))
        elif n.op in {"add", "mul", "div"} and len(n.inputs) >= 2:
            a_shape = shapes.get(n.inputs[0], tuple())
            b_shape = shapes.get(n.inputs[1], tuple())
            shape = tuple(_to_int_dim(d) for d in _public_broadcast_shape_tuple(a_shape, b_shape))
        elif n.op == "broadcast" and len(n.inputs) >= 2:
            shape = tuple(_to_int_dim(d) for d in shapes.get(n.inputs[1], tuple()))
        elif n.op == "matmul" and len(n.inputs) >= 2:
            a_shape = shapes.get(n.inputs[0], tuple())
            b_shape = shapes.get(n.inputs[1], tuple())
            if len(a_shape) == 2 and len(b_shape) == 2:
                shape = (_to_int_dim(a_shape[0]), _to_int_dim(b_shape[1]))
            else:
                shape = tuple(_to_int_dim(d) for d in a_shape)
        elif n.op == "transpose" and n.inputs:
            in_shape = shapes.get(n.inputs[0], tuple())
            if len(in_shape) == 2:
                shape = (_to_int_dim(in_shape[1]), _to_int_dim(in_shape[0]))
            else:
                shape = tuple(_to_int_dim(d) for d in in_shape)
        elif n.op in {"sqrt", "exp", "relu", "silu"} and n.inputs:
            shape = tuple(_to_int_dim(d) for d in shapes.get(n.inputs[0], tuple()))
        elif n.inputs:
            shape = tuple(_to_int_dim(d) for d in shapes.get(n.inputs[0], tuple()))
        else:
            shape = tuple(_to_int_dim(d) for d in n.attrs.get("shape", n.shape or tuple()))
        n.shape = shape
        shapes[n.id] = shape
    return G


def _node_by_id(G: nuGraph, node_id: str) -> Optional[Node]:
    for n in G.nodes:
        if n.id == node_id:
            return n
    return None


def _position_by_id(G: nuGraph, node_id: str) -> Optional[int]:
    for i, n in enumerate(G.nodes):
        if n.id == node_id:
            return i
    return None


def _immediate_successor_positions(G: nuGraph, pos: int) -> list[int]:
    node_id = G.node_at(pos).id
    return [i for i, n in enumerate(G.nodes) if node_id in n.inputs]


def _effective_input_ids(G: nuGraph, pos: int) -> list[str]:
    return list(dict.fromkeys(G.node_at(pos).inputs))


def _swap_with_successor(
    G_cur: nuGraph, pos: int, succ_pos: int, clone_inputs: set[str]
) -> Optional[tuple[nuGraph, int]]:
    op1 = G_cur.node_at(pos)
    op2 = G_cur.node_at(succ_pos)
    if op1.op not in {"mul", "div"} or op2.op != "matmul":
        return None
    if len(op1.inputs) != 2 or len(op2.inputs) != 2 or op2.inputs[0] != op1.id:
        return None
    if len(G_cur.successors(op1)) != 1:
        return None
    if clone_inputs != {op1.inputs[0]}:
        return None

    G_new = G_cur.clone()
    n1 = _node_by_id(G_new, op1.id)
    n2 = _node_by_id(G_new, op2.id)
    if n1 is None or n2 is None:
        return None

    data_id, scale_id = n1.inputs
    w_id = n2.inputs[1]
    n1.op = "matmul"
    n1.inputs = [data_id, w_id]
    n1.attrs = {}
    n2.op = op1.op
    n2.inputs = [n1.id, scale_id]
    n2.attrs = dict(op1.attrs)
    annotate_shapes_concrete(G_new)
    new_pos = _position_by_id(G_new, n2.id)
    if new_pos is None:
        return None
    return G_new, new_pos


def z3_equivalent_order(
    op1: Node, op2: Node, G_cur: nuGraph, G_new: Optional[nuGraph] = None, verbose: bool = False
) -> bool:
    forbidden = {
        ("reduce_sum", "broadcast"),
        ("broadcast", "reduce_sum"),
        ("reduce_sum", "sqrt"),
        ("sqrt", "reduce_sum"),
    }
    if (op1.op, op2.op) in forbidden:
        return False

    if op1.op in {"mul", "div"} and op2.op == "matmul":
        annotate_shapes_concrete(G_cur)
        lhs = _node_by_id(G_cur, op1.id)
        rhs = _node_by_id(G_cur, op2.id)
        if lhs is None or rhs is None or len(lhs.inputs) != 2 or len(rhs.inputs) != 2:
            return False
        if rhs.inputs[0] != lhs.id:
            return False
        shape_map = {n.id: n.shape or tuple() for n in G_cur.nodes}
        x_shape = shape_map.get(lhs.inputs[0], tuple())
        scale_shape = shape_map.get(lhs.inputs[1], tuple())
        if len(x_shape) != 2 or len(scale_shape) != 2:
            return False
        return _dims_equal(scale_shape[0], x_shape[0]) and _dims_equal(scale_shape[1], 1)

    return False




# COMPLETE THIS FUNCTION
def nu_graph_generation_z3(G : nuGraph, verbose=False) -> List[nuGraph]:
    G0 = annotate_shapes_concrete(G.clone())
    M: set[nuGraph] = {G0}

    for op1_orig in [n for n in G0.nodes if n.op != "input"]:
        M_next: set[nuGraph] = set()
        for G_seed in sorted(M, key=graph_signature):
            pos0 = _position_by_id(G_seed, op1_orig.id)
            if pos0 is None:
                M_next.add(G_seed)
                continue

            worklist: list[tuple[nuGraph, int]] = [(G_seed, pos0)]
            visited: set[tuple[str, int]] = set()

            while worklist:
                G_cur, pos = worklist.pop()
                state_key = (graph_signature(G_cur), pos)
                if state_key in visited:
                    continue
                visited.add(state_key)
                advanced = False

                for succ_pos in _immediate_successor_positions(G_cur, pos):
                    op1 = G_cur.node_at(pos)
                    op2 = G_cur.node_at(succ_pos)
                    inputs = _effective_input_ids(G_cur, pos)
                    accepted = False

                    for k in range(1, len(inputs) + 1):
                        for subset in combinations(inputs, k):
                            swapped = _swap_with_successor(G_cur, pos, succ_pos, set(subset))
                            if swapped is None:
                                continue
                            G_new, p_new = swapped
                            if not z3_equivalent_order(op1, op2, G_cur, G_new, verbose=verbose):
                                continue
                            M_next.add(G_new)
                            worklist.append((G_new, p_new))
                            advanced = True
                            accepted = True
                            break
                        if accepted:
                            break

                if not advanced:
                    M_next.add(G_cur)

        M |= M_next

    out = sorted(M, key=graph_signature)
    for gv in out:
        annotate_shapes_concrete(gv)
    return out






def kernel_matmul_red_div(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    rec = y.sum(axis=1, keep_dims=True)
    return (x / rec) @ w


def kernel_matmul_red_mul(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    rec = y.sum(axis=1, keep_dims=True)
    return (x * rec) @ w


def kernel_broadcast_row_bias_add(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    bias = y.sum(axis=1, keep_dims=True)
    bias_b = bias.broadcast_like(x)
    z = x + bias_b
    return z @ w


def kernel_reduce_mul_broadcast(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    rec = y.sum(axis=1, keep_dims=True)
    z = x * rec
    z_b = z.broadcast_like(x)
    return z_b @ w


def kernel_reduce_broadcast_mul(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    rec = y.sum(axis=1, keep_dims=True)
    rec_b = rec.broadcast_like(x)
    z = x * rec_b
    return z @ w


def kernel_rmsnorm_matmul(x: AxonArray, y: AxonArray, w: AxonArray) -> AxonArray:
    yy = y * y
    rec = yy.sum(axis=1, keep_dims=True)
    rms = rec.sqrt()
    norm = x / rms
    return norm @ w
  

def kernel_softmax_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    ex = x.exp()
    den = ex.sum(axis=1, keep_dims=True)
    probs = ex / den
    return probs @ w


def kernel_transpose_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    xt = x.transpose()
    return xt @ w


def kernel_relu_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    return x.relu() @ w


def kernel_silu_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    return x.silu() @ w


def _base_inputs(M: int, K: int, N: int) -> list[Node]:
    return [
        Node(id="x", op="input", inputs=[], attrs={"shape": (M, K)}),
        Node(id="y", op="input", inputs=[], attrs={"shape": (M, K)}),
        Node(id="w", op="input", inputs=[], attrs={"shape": (K, N)}),
    ]


def build_kernel_matmul_red_div_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_matmul_red_div(AxonArray("x", (M, K)), AxonArray("y", (M, K)), AxonArray("w", (K, N)))
    G = nuGraph(_base_inputs(M, K, N) + [
        Node(id="rec", op="reduce_sum", inputs=["y"], attrs={"axis": 1, "keep_dims": True}),
        Node(id="scale", op="div", inputs=["x", "rec"], attrs={}),
        Node(id="out", op="matmul", inputs=["scale", "w"], attrs={}),
    ])
    annotate_shapes_concrete(G)
    return G


def build_kernel_matmul_red_mul_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_matmul_red_mul(AxonArray("x", (M, K)), AxonArray("y", (M, K)), AxonArray("w", (K, N)))
    G = nuGraph(_base_inputs(M, K, N) + [
        Node(id="rec", op="reduce_sum", inputs=["y"], attrs={"axis": 1, "keep_dims": True}),
        Node(id="scale", op="mul", inputs=["x", "rec"], attrs={}),
        Node(id="out", op="matmul", inputs=["scale", "w"], attrs={}),
    ])
    annotate_shapes_concrete(G)
    return G


def build_kernel_broadcast_row_bias_add_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_broadcast_row_bias_add(AxonArray("x", (M, K)), AxonArray("y", (M, K)), AxonArray("w", (K, N)))
    G = nuGraph(_base_inputs(M, K, N) + [
        Node(id="bias", op="reduce_sum", inputs=["y"], attrs={"axis": 1, "keep_dims": True}),
        Node(id="bias_b", op="broadcast", inputs=["bias", "x"], attrs={}),
        Node(id="z", op="add", inputs=["x", "bias_b"], attrs={}),
        Node(id="out", op="matmul", inputs=["z", "w"], attrs={}),
    ])
    annotate_shapes_concrete(G)
    return G


def build_kernel_reduce_mul_broadcast_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_reduce_mul_broadcast(AxonArray("x", (M, K)), AxonArray("y", (M, K)), AxonArray("w", (K, N)))
    G = nuGraph(_base_inputs(M, K, N) + [
        Node(id="rec", op="reduce_sum", inputs=["y"], attrs={"axis": 1, "keep_dims": True}),
        Node(id="z", op="mul", inputs=["x", "rec"], attrs={}),
        Node(id="z_b", op="broadcast", inputs=["z", "x"], attrs={}),
        Node(id="out", op="matmul", inputs=["z_b", "w"], attrs={}),
    ])
    annotate_shapes_concrete(G)
    return G


def build_kernel_reduce_broadcast_mul_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_reduce_broadcast_mul(AxonArray("x", (M, K)), AxonArray("y", (M, K)), AxonArray("w", (K, N)))
    G = nuGraph(_base_inputs(M, K, N) + [
        Node(id="rec", op="reduce_sum", inputs=["y"], attrs={"axis": 1, "keep_dims": True}),
        Node(id="rec_b", op="broadcast", inputs=["rec", "x"], attrs={}),
        Node(id="z", op="mul", inputs=["x", "rec_b"], attrs={}),
        Node(id="out", op="matmul", inputs=["z", "w"], attrs={}),
    ])
    annotate_shapes_concrete(G)
    return G


def build_kernel_rmsnorm_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_rmsnorm_matmul(
        AxonArray("x", (M, K)),
        AxonArray("y", (M, K)),
        AxonArray("w", (K, N)),
    )
    nodes = _base_inputs(M, K, N) + [
        Node(id="yy", op="mul", inputs=["y", "y"], attrs={}),
        Node(id="rec", op="reduce_sum", inputs=["yy"], attrs={"axis": 1, "keep_dims": True}),
        Node(id="rms", op="sqrt", inputs=["rec"], attrs={}),
        Node(id="norm", op="div", inputs=["x", "rms"], attrs={}),
        Node(id="out", op="matmul", inputs=["norm", "w"], attrs={}),
    ]
    G = nuGraph(nodes)
    annotate_shapes_concrete(G)
    return G


def build_kernel_softmax_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_softmax_matmul(
        AxonArray("x", (M, K)),
        AxonArray("w", (K, N)),
    )
    nodes = [
        Node(id="x", op="input", inputs=[], attrs={"shape": (M, K)}),
        Node(id="w", op="input", inputs=[], attrs={"shape": (K, N)}),
        Node(id="ex", op="exp", inputs=["x"], attrs={}),                                       # (M,K)
        Node(id="den", op="reduce_sum", inputs=["ex"], attrs={"axis": 1, "keep_dims": True}),  # (M,1)
        Node(id="probs", op="div", inputs=["ex", "den"], attrs={}),                             # (M,K)
        Node(id="out", op="matmul", inputs=["probs", "w"], attrs={}),                           # (M,N)
    ]
    G = nuGraph(nodes)
    annotate_shapes_concrete(G)
    return G


def build_kernel_transpose_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_transpose_matmul(
        AxonArray("x", (M, K)),
        AxonArray("w", (M, N)),
    )
    nodes = [
        Node(id="x", op="input", inputs=[], attrs={"shape": (M, K)}),
        Node(id="w", op="input", inputs=[], attrs={"shape": (M, N)}),
        Node(id="xt", op="transpose", inputs=["x"], attrs={}),       # (K,M)
        Node(id="out", op="matmul", inputs=["xt", "w"], attrs={}),   # (K,N)
    ]
    G = nuGraph(nodes)
    annotate_shapes_concrete(G)
    return G


def build_kernel_relu_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_relu_matmul(AxonArray("x", (M, K)), AxonArray("w", (K, N)))
    G = nuGraph([
        Node(id="x", op="input", inputs=[], attrs={"shape": (M, K)}),
        Node(id="w", op="input", inputs=[], attrs={"shape": (K, N)}),
        Node(id="z", op="relu", inputs=["x"], attrs={}),
        Node(id="out", op="matmul", inputs=["z", "w"], attrs={}),
    ])
    annotate_shapes_concrete(G)
    return G

def build_kernel_silu_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    _ = kernel_silu_matmul(AxonArray("x", (M, K)), AxonArray("w", (K, N)))
    G = nuGraph([
        Node(id="x", op="input", inputs=[], attrs={"shape": (M, K)}),
        Node(id="w", op="input", inputs=[], attrs={"shape": (K, N)}),
        Node(id="z", op="silu", inputs=["x"], attrs={}),
        Node(id="out", op="matmul", inputs=["z", "w"], attrs={}),
    ])
    annotate_shapes_concrete(G)
    return G


def print_graph(G: nuGraph) -> None:
    for i, n in enumerate(G.nodes):
        print(f"[{i}] id={n.id:12s} op={n.op:10s} inputs={n.inputs} shape={n.shape} attrs={n.attrs}")


def _variants_for(builder: Callable[[int, int, int], nuGraph], M: int = 4, K: int = 8, N: int = 16) -> list[nuGraph]:
    G0 = builder(M, K, N)
    return nu_graph_generation_z3(G0, verbose=False)


def _test_expected_variant_counts() -> None:
    cases: list[tuple[str, Callable[[int, int, int], nuGraph], int]] = [
        ("kernel_matmul_red_div", build_kernel_matmul_red_div_graph, 1),
        ("kernel_matmul_red_mul", build_kernel_matmul_red_mul_graph, 1),
        ("kernel_broadcast_row_bias_add", build_kernel_broadcast_row_bias_add_graph, 1),
        ("kernel_reduce_mul_broadcast", build_kernel_reduce_mul_broadcast_graph, 1),
        ("kernel_reduce_broadcast_mul", build_kernel_reduce_broadcast_mul_graph, 1),
    ]

    for name, builder, expected_min in cases:
        vs = _variants_for(builder)
        got = len(vs)
        assert got >= expected_min, f"{name}: expected >= {expected_min} variants, got {got}"
        print(f" {name}: variants={got} (expected >= {expected_min})")


def _test_no_illegal_reduce_broadcast_swap() -> None:
    G = build_kernel_reduce_broadcast_mul_graph(4, 8, 16)
    rec = next(n for n in G.nodes if n.id == "rec")
    rec_b = next(n for n in G.nodes if n.id == "rec_b")
    ok = z3_equivalent_order(rec, rec_b, G, verbose=False)
    assert not ok, "reduce_sum <-> broadcast must be rejected"
    print(" reduce_sum<->broadcast illegal swap rejected")


def _test_no_illegal_reduce_sqrt_swap() -> None:
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)
    rec = next(n for n in G.nodes if n.id == "rec")
    rms = next(n for n in G.nodes if n.id == "rms")
    ok = z3_equivalent_order(rec, rms, G, verbose=False)
    assert not ok, "reduce_sum <-> sqrt must be rejected by axioms"
    print(" reduce_sum<->sqrt illegal swap rejected (axiomatic)")


def _test_rmsnorm_matmul_graph() -> None:
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)

    yy = next(n for n in G.nodes if n.id == "yy")
    rec = next(n for n in G.nodes if n.id == "rec")
    rms = next(n for n in G.nodes if n.id == "rms")
    norm = next(n for n in G.nodes if n.id == "norm")
    out = next(n for n in G.nodes if n.id == "out")

    assert yy.shape == (4, 8), f"yy shape wrong: {yy.shape}"
    assert rec.shape == (4, 1), f"rec shape wrong: {rec.shape}"
    assert rms.shape == (4, 1), f"rms shape wrong: {rms.shape}"
    assert norm.shape == (4, 8), f"norm shape wrong: {norm.shape}"
    assert out.shape == (4, 16), f"out shape wrong: {out.shape}"

    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 1, "rmsnorm_matmul should emit at least org variant"
    print(" rmsnorm_matmul (+sqrt) graph builds and runs variant generation")


def _test_softmax_matmul_graph() -> None:
    G = build_kernel_softmax_matmul_graph(4, 8, 16)
    ex = next(n for n in G.nodes if n.id == "ex")
    den = next(n for n in G.nodes if n.id == "den")
    probs = next(n for n in G.nodes if n.id == "probs")
    out = next(n for n in G.nodes if n.id == "out")

    assert ex.shape == (4, 8), f"ex shape wrong: {ex.shape}"
    assert den.shape == (4, 1), f"den shape wrong: {den.shape}"
    assert probs.shape == (4, 8), f"probs shape wrong: {probs.shape}"
    assert out.shape == (4, 16), f"out shape wrong: {out.shape}"

    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 1, "softmax_matmul should emit at least org variant"
    print(" softmax_matmul graph builds and runs variant generation")


def _test_transpose_matmul_graph() -> None:
    G = build_kernel_transpose_matmul_graph(4, 8, 16)
    xt = next(n for n in G.nodes if n.id == "xt")
    out = next(n for n in G.nodes if n.id == "out")

    assert xt.shape == (8, 4), f"xt shape wrong: {xt.shape}"
    assert out.shape == (8, 16), f"out shape wrong: {out.shape}"

    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 1, "transpose_matmul should emit at least org variant"
    print(" transpose_matmul graph builds and runs variant generation")


def _test_relu_matmul_graph() -> None:
    G = build_kernel_relu_matmul_graph(4, 8, 16)
    z = next(n for n in G.nodes if n.id == "z")
    out = next(n for n in G.nodes if n.id == "out")
    assert z.shape == (4, 8)
    assert out.shape == (4, 16)
    assert len(nu_graph_generation_z3(G, verbose=False)) >= 1

def _test_silu_matmul_graph() -> None:
    G = build_kernel_silu_matmul_graph(4, 8, 16)
    z = next(n for n in G.nodes if n.id == "z")
    out = next(n for n in G.nodes if n.id == "out")
    assert z.shape == (4, 8)
    assert out.shape == (4, 16)
    assert len(nu_graph_generation_z3(G, verbose=False)) >= 1


def _test_axonarray_ops_shapes() -> None:
    x = AxonArray("x", (4, 8))
    y = AxonArray("y", (4, 8))
    s = y.sum(axis=1, keep_dims=True)
    assert s.shape == (4, 1)
    assert (x * s).shape == (4, 8)
    assert (x / s).shape == (4, 8)
    assert x.broadcast_like(y).shape == (4, 8)
    assert x.transpose().shape == (8, 4)
    assert x.relu().shape == (4, 8)
    assert x.silu().shape == (4, 8)
    assert x.exp().shape == (4, 8)


def _test_mul_div_matmul_variant_growth() -> None:
    v_mul = _variants_for(build_kernel_matmul_red_mul_graph)
    v_div = _variants_for(build_kernel_matmul_red_div_graph)
    assert len(v_mul) >= 2
    assert len(v_div) >= 2


def run_all_tests() -> None:
    print("\n================ RUNNING NU-GRAPH TESTS ================")
    _test_compositional_shape_equivalence()
    # _test_expected_variant_counts()
    # _test_no_illegal_reduce_broadcast_swap()
    # _test_no_illegal_reduce_sqrt_swap()
    # _test_rmsnorm_matmul_graph()
    # _test_softmax_matmul_graph()
    # _test_transpose_matmul_graph()
    _test_relu_matmul_graph()
    _test_silu_matmul_graph()
    print("=============== ALL TESTS PASSED  =====================\n")


if __name__ == "__main__":
    #run_all_tests()

    kernels: list[tuple[str, Callable[[int, int, int], nuGraph]]] = [
        ("kernel_matmul_red_div", build_kernel_matmul_red_div_graph),
        ("kernel_matmul_red_mul", build_kernel_matmul_red_mul_graph),
        # ("kernel_broadcast_row_bias_add", build_kernel_broadcast_row_bias_add_graph),
        # ("kernel_reduce_mul_broadcast", build_kernel_reduce_mul_broadcast_graph),
        # ("kernel_reduce_broadcast_mul", build_kernel_reduce_broadcast_mul_graph),
        # ("kernel_rmsnorm_matmul", build_kernel_rmsnorm_matmul_graph),
        # ("kernel_softmax_matmul", build_kernel_softmax_matmul_graph),
        # ("kernel_transpose_matmul", build_kernel_transpose_matmul_graph),
        # ("kernel_relu_matmul", build_kernel_relu_matmul_graph),
        # ("kernel_silu_matmul", build_kernel_silu_matmul_graph),
    ]

    for kname, builder in kernels:
        print("\n" + "=" * 80)
        print(f"Tracing kernel: {kname}")
        print("=" * 80)
        G0 = builder(4, 8, 16)
        variants = nu_graph_generation_z3(G0, verbose=True)

        print(f"Found {len(variants)} variant(s) for {kname}\n")
        for vi, gv in enumerate(variants):
            print(f"=== {kname} :: Variant {vi} ===")
            print_graph(gv)
            print()
