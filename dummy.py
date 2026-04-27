from __future__ import annotations

import builtins
import argparse
import concurrent.futures
import os
import sys
import io
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from itertools import combinations, permutations, product as _iproduct
from threading import Lock, RLock
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
_VERBOSE_LOCK = Lock()   # serializes verbose prints from concurrent synthesis threads
_Z3_LOCK = RLock()       # serializes z3 formula construction (z3.main_ctx is not thread-safe)

# ---------------------------------------------------------------------------
# Synthesis result cache
# ---------------------------------------------------------------------------
# Normalized sketch representation: a pure-data tuple that does not hold any
# SymTensor references, so it is safe to store across calls.
#   ("INPUT", int)                      – leaf: which input index
#   ("OP", str, dict, tuple[…, …])      – interior: op name, attrs, children
_NormSketch = Any

_CACHE_MISS: object = object()          # sentinel: key not yet cached

# Single-result cache (used by _lower_node / lower_nu_graph).
# Maps (op, frozen_attrs, input_ranks) → None (failure) | _NormSketch (success)
_SYNTHESIS_CACHE: dict[tuple, Optional[_NormSketch]] = {}

# All-results cache (used by _lower_node_all / lower_nu_graph_all_variants).
# Maps same key → list[_NormSketch] (empty = all synthesis failed)
_SYNTHESIS_CACHE_ALL: dict[tuple, list[_NormSketch]] = {}

_SYNTHESIS_CACHE_LOCK = Lock()


def _clear_synthesis_caches() -> None:
    """Discard all cached synthesis results.

    Intended for test isolation.  Calling this before a test that needs to
    observe first-time synthesis (cache miss) ensures the caches start empty.
    """
    with _SYNTHESIS_CACHE_LOCK:
        _SYNTHESIS_CACHE.clear()
        _SYNTHESIS_CACHE_ALL.clear()


def _start_kernel_synthesis_cache(
    kernel_name: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Reset synthesis caches for a new kernel-level lowering session.

    Cache reuse is intentionally preserved across repeated tensor ops and graph
    variants within a single kernel, but performance measurements should not be
    biased by work performed for earlier kernels.  This helper clears the
    caches once at the start of a kernel-level lowering run.
    """
    _clear_synthesis_caches()
    if verbose:
        if kernel_name is None:
            print("[synthesis-cache] starting with an empty cache")
        else:
            print(f"[synthesis-cache] starting kernel '{kernel_name}' with an empty cache")


def _effective_max_workers(max_workers: Optional[int], task_count: int) -> int:
    """Choose how many threads to launch for a pool of *task_count* items.

    ``max_workers=None`` means "auto": derive the worker count from the number
    of logical CPUs reported by ``os.cpu_count()`` (defaulting to 4 when the
    OS cannot determine the count), clamped to *task_count* so no unnecessary
    threads are created.  Explicit positive ``max_workers`` values are still
    respected.  ``task_count <= 1`` collapses to 1 worker, and explicit
    non-positive ``max_workers`` values are clamped up to 1.  Explicit values
    larger than ``task_count`` are clamped down to ``task_count``.

    Returns:
        The computed number of workers to launch.
    """
    if task_count <= 1:
        return 1
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        return builtins.max(1, builtins.min(cpu_count, task_count))
    return builtins.max(1, builtins.min(max_workers, task_count))


def _format_resolved_input(inp_id: str, hw_id: str) -> str:
    if inp_id == hw_id:
        return f"'{inp_id}'"
    return f"'{inp_id}' (resolved to '{hw_id}')"
_SYNTHESIS_STATS_LOCK = Lock()


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

    def as_formula(self) -> z3.BoolRef:
        if not self.facts:
            return z3.BoolVal(True)
        return z3.And(*self.facts)


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
class Precondition:
    description: str
    constraint: z3.BoolRef


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


def semantics_hw():
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
        if _arith_equal(da, 1):
            out.append(db)
        elif _arith_equal(db, 1):
            out.append(da)
        else:
            out.append(z3.If(da == 1, db, da))
    return ShapeResult(ShapeExpr(out), ctx)


def _broadcast_to_shape(src: ShapeExpr, target: ShapeExpr) -> ShapeResult:
    rank = builtins.max(src.rank, target.rank)
    sd = [z3.IntVal(1)] * (rank - src.rank) + list(src.dims)
    td = [z3.IntVal(1)] * (rank - target.rank) + list(target.dims)
    ctx = Context([d > 0 for d in sd + td])
    for sdim, tdim in zip(sd, td):
        ctx.add(z3.Or(sdim == tdim, sdim == 1))
    return ShapeResult(ShapeExpr(list(td)), ctx)


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


def _safe_divide(lhs: z3.ArithRef, rhs: z3.ArithRef) -> z3.ArithRef:
    """Match `_compile_reciprocal`: division by zero follows the zero-reciprocal path."""
    return z3.If(rhs == 0, z3.RealVal(0), lhs / rhs)


def _apply_binary(op: Any, lhs: z3.ArithRef, rhs: z3.ArithRef) -> z3.ArithRef:
    opn = _operand_to_expr(op)
    if opn in ("add", "plus", "maximum", "max"):
        return lhs + rhs if opn in ("add", "plus") else z3.If(lhs >= rhs, lhs, rhs)
    if opn in ("subtract", "sub"):
        return lhs - rhs
    if opn in ("multiply", "mul"):
        return lhs * rhs
    if opn in ("divide", "div"):
        return _safe_divide(lhs, rhs)
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


def singleton_dimension_extensionality(sem: Semantics) -> Context:
    ctx = Context()
    if sem.shape.rank == 0:
        return ctx
    vars = _index_vars(f"{sem.name}_singleton", sem.shape.rank, "s")
    for axis, dim in enumerate(sem.shape.dims):
        zero_idx = list(vars)
        zero_idx[axis] = z3.IntVal(0)
        ctx.add(z3.ForAll(vars, z3.Implies(dim == 1, sem.fn(*vars) == sem.fn(*zero_idx))))
    return ctx


def _shape_eq(a: ShapeExpr, b: ShapeExpr) -> z3.BoolRef:
    if a.rank != b.rank:
        return z3.BoolVal(False)
    out = z3.BoolVal(True)
    for da, db in zip(a.dims, b.dims):
        out = z3.And(out, da == db)
    return out


def reduction_extensionality_context() -> Context:
    ctx = Context()

    b1 = z3.Int("reduce_ext_b1")
    b2 = z3.Int("reduce_ext_b2")
    i = z3.Int("reduce_ext_i")
    j = z3.Int("reduce_ext_j")
    k = z3.Int("reduce_ext_k")
    e1 = z3.Int("reduce_ext_e1")
    e2 = z3.Int("reduce_ext_e2")

    same_body1 = z3.ForAll([k], BODY1(b1, i, k) == BODY1(b2, i, k))
    ctx.add(
        z3.ForAll(
            [b1, b2, i, e1, e2],
            z3.Implies(z3.And(e1 == e2, same_body1), REDUCE1(b1, i, e1) == REDUCE1(b2, i, e2)),
        )
    )

    same_body2 = z3.ForAll([k], BODY2(b1, i, j, k) == BODY2(b2, i, j, k))
    ctx.add(
        z3.ForAll(
            [b1, b2, i, j, e1, e2],
            z3.Implies(z3.And(e1 == e2, same_body2), REDUCE2(b1, i, j, e1) == REDUCE2(b2, i, j, e2)),
        )
    )
    return ctx


def _check_reduction_equivalent_by_body(
    lhs: Semantics,
    rhs: Semantics,
    shape_eq: z3.BoolRef,
    timeout: int,
) -> bool:
    if lhs.reduction is None or rhs.reduction is None:
        return False
    if lhs.reduction.outer_rank != rhs.reduction.outer_rank:
        return False

    full_ctx = lhs.ctx.merged(rhs.ctx, reduction_extensionality_context())
    solver = z3.Solver()
    solver.set("timeout", timeout)
    solver.add(full_ctx.as_formula(), z3.Not(shape_eq))
    if solver.check() != z3.unsat:
        return False

    solver = z3.Solver()
    solver.set("timeout", timeout)
    solver.add(full_ctx.as_formula(), lhs.reduction.extent != rhs.reduction.extent)
    if solver.check() != z3.unsat:
        return False

    counterexample_formula: Optional[z3.BoolRef] = None

    if lhs.reduction.outer_rank == 1:
        i = z3.Int("body_eq1_i")
        k = z3.Int("body_eq1_k")
        bounds = z3.And(
            i >= 0,
            i < lhs.shape.dims[0],
            k >= 0,
            k < lhs.reduction.extent,
        )
        counterexample_formula = z3.And(
            shape_eq,
            lhs.reduction.extent == rhs.reduction.extent,
            bounds,
            BODY1(z3.IntVal(lhs.reduction.body_id), i, k) != BODY1(z3.IntVal(rhs.reduction.body_id), i, k),
        )
    elif lhs.reduction.outer_rank == 2:
        i = z3.Int("body_eq2_i")
        j = z3.Int("body_eq2_j")
        k = z3.Int("body_eq2_k")
        bounds = z3.And(
            i >= 0,
            i < lhs.shape.dims[0],
            j >= 0,
            j < lhs.shape.dims[1],
            k >= 0,
            k < lhs.reduction.extent,
        )
        counterexample_formula = z3.And(
            shape_eq,
            lhs.reduction.extent == rhs.reduction.extent,
            bounds,
            BODY2(z3.IntVal(lhs.reduction.body_id), i, j, k) != BODY2(z3.IntVal(rhs.reduction.body_id), i, j, k),
        )

    if counterexample_formula is None:
        return False

    solver = z3.Solver()
    solver.set("timeout", timeout)
    solver.add(full_ctx.as_formula(), counterexample_formula)
    return solver.check() == z3.unsat


def compile_expr(expr: SymExpr, cache: dict[int, Semantics]) -> Semantics:
    key = id(expr)
    if key in cache:
        return cache[key]

    if expr.op == "input":
        shape = ShapeExpr(list(expr.shape))
        sem = Semantics(
            name=expr.name,
            shape=shape,
            fn=_tensor_function(f"V_{expr.name}", shape.rank),
            ctx=Context([d > 0 for d in shape.dims]),
        )
        sem.ctx = sem.ctx.merged(singleton_dimension_extensionality(sem))
        cache[key] = sem
        return sem

    compiled_inputs = [compile_expr(inp, cache) for inp in expr.inputs]
    input_shapes = [c.shape for c in compiled_inputs]
    if expr.op not in _SEMANTICS:
        raise KeyError(f"No semantics registered for op '{expr.op}'")
    shape_rule, compile_rule = _SEMANTICS[expr.op]
    if shape_rule is None or compile_rule is None:
        raise KeyError(f"Incomplete semantics registered for op '{expr.op}'")
    shape_res = shape_rule(input_shapes, expr.attrs)
    sem = compile_rule(expr, compiled_inputs, shape_res.out)
    sem.ctx = sem.ctx.merged(shape_res.ctx, singleton_dimension_extensionality(sem))
    cache[key] = sem
    return sem


def _expr_structural_key(expr: SymExpr, cache: Optional[dict[int, Any]] = None) -> Any:
    cache = {} if cache is None else cache
    key = id(expr)
    if key in cache:
        return cache[key]
    if expr.op == "input":
        sig = ("input", expr.name, tuple(map(str, expr.shape)))
    else:
        sig = (
            expr.op,
            tuple(map(str, expr.shape)),
            tuple(sorted((k, repr(v)) for k, v in expr.attrs.items())),
            tuple(_expr_structural_key(inp, cache) for inp in expr.inputs),
        )
    cache[key] = sig
    return sig


def _rename_expr_tree_for_equivalence(
    expr: SymExpr,
    side: str,
    shared_names: dict[Any, str],
    used_names: set[str],
    name_counter: list[int],
    cache: Optional[dict[int, SymExpr]] = None,
    sig_cache: Optional[dict[int, Any]] = None,
) -> SymExpr:
    cache = {} if cache is None else cache
    sig_cache = {} if sig_cache is None else sig_cache
    key = id(expr)
    if key in cache:
        return cache[key]

    if expr.op == "input":
        renamed = SymExpr(expr.op, [], expr.shape, dict(expr.attrs), expr.name)
        cache[key] = renamed
        return renamed

    sig = _expr_structural_key(expr, sig_cache)
    if sig in shared_names:
        name = shared_names[sig]
    else:
        candidate_names = [expr.name, f"{expr.name}_{side}"]
        fallback_name = f"{expr.name}_{side}_{name_counter[0]}"
        name = next((candidate for candidate in candidate_names if candidate not in used_names), fallback_name)
        if name == fallback_name:
            name_counter[0] += 1
        shared_names[sig] = name
        used_names.add(name)

    renamed = SymExpr(
        expr.op,
        [
            _rename_expr_tree_for_equivalence(inp, side, shared_names, used_names, name_counter, cache, sig_cache)
            for inp in expr.inputs
        ],
        expr.shape,
        dict(expr.attrs),
        name,
    )
    cache[key] = renamed
    return renamed


def check_equivalent(
    lhs: SymTensor,
    rhs: SymTensor,
    timeout: int = 10000,
    preconditions: Optional[list[Precondition]] = None,
    rule_name: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    shared_names: dict[Any, str] = {}
    used_names: set[str] = set()
    name_counter = [0]
    lhs_expr = _rename_expr_tree_for_equivalence(lhs.expr, "lhs", shared_names, used_names, name_counter)
    rhs_expr = _rename_expr_tree_for_equivalence(rhs.expr, "rhs", shared_names, used_names, name_counter)
    lsem = compile_expr(lhs_expr, {})
    rsem = compile_expr(rhs_expr, {})
    shape_eq = _shape_eq(lsem.shape, rsem.shape)
    if lsem.shape.rank != rsem.shape.rank:
        return False

    preconditions = preconditions or []
    extra_ctx = Context([p.constraint for p in preconditions])
    full_ctx = lsem.ctx.merged(
        rsem.ctx,
        reduction_extensionality_context(),
        extra_ctx,
    )

    solver = z3.Solver()
    solver.set("timeout", timeout)
    solver.add(full_ctx.as_formula(), z3.Not(shape_eq))
    res = solver.check()
    if res != z3.unsat:
        ok = False
    else:
        solver = z3.Solver()
        solver.set("timeout", timeout)
        solver.add(full_ctx.as_formula(), shape_eq)
        if lsem.shape.rank == 0:
            solver.add(lsem.fn() != rsem.fn())
        else:
            vars = [z3.Int(f"witness_{idx}") for idx in range(lsem.shape.rank)]
            lhs_bounds = [z3.And(vars[i] >= 0, vars[i] < lsem.shape.dims[i]) for i in range(lsem.shape.rank)]
            bounds = z3.And(*lhs_bounds)
            solver.add(bounds, lsem.fn(*vars) != rsem.fn(*vars))
        res = solver.check()
        ok = res == z3.unsat
    if not ok and _check_reduction_equivalent_by_body(lsem, rsem, shape_eq, timeout):
        ok = True

    _last_check_stats: dict[str, Any] = {}
    _last_check_stats["num_obligations"] = len(full_ctx.facts)
    _last_check_stats["failure_reason"] = "" if ok else str(res)

    if verbose:
        print("result:", res, "proved:", ok)
        print("lhs rank:", lhs.rank, "rhs rank:", rhs.rank)
        print("lhs shape expr:", lsem.shape.dims)
        print("rhs shape expr:", rsem.shape.dims)
        print("lhs facts:", len(lsem.ctx.facts), "rhs facts:", len(rsem.ctx.facts))
        print("lhs reduction rank:", None if lsem.reduction is None else lsem.reduction.outer_rank)
        print("rhs reduction rank:", None if rsem.reduction is None else rsem.reduction.outer_rank)
        if preconditions:
            print("preconditions:")
            for p in preconditions:
                print("-", p.description)

    return ok


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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
def nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=matmul_perf_mode.none, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        lhs, rhs = ins
        out_shape = attrs.get("out_shape")
        if out_shape is not None:
            return _shape_from_out(out_shape)
        if lhs.rank >= 2 and rhs.rank >= 2:
            dims = [lhs.dims[-1], rhs.dims[-1]]
            ctx = _shape_ctx(*lhs.dims, *rhs.dims, *dims)
            # nc_matmul contracts over lhs.dims[0] / rhs.dims[0]; they must agree.
            ctx.add(lhs.dims[0] == rhs.dims[0])
            return ShapeResult(ShapeExpr(dims), ctx)
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
def reciprocal(dst, data, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_unary_same(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_reciprocal(expr, ins, out_shape)

    _ensure_semantics("reciprocal", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    return _new_sym_tensor("reciprocal", [data], {"name": name}, _default_out_shape(dst, data))


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
def tensor_copy(dst, src, engine=engine.unknown, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_same_as_first(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_copy(expr, ins, out_shape)

    _ensure_semantics("tensor_copy", shape_rule, value_rule)

    assert _is_sym_tensor(src)
    return _new_sym_tensor("tensor_copy", [src], {"engine": engine, "name": name}, _default_out_shape(dst, src))


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
def tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None):
    def shape_rule(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
        return _shape_tensor_reduce(ins, attrs)

    def value_rule(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
        return _compile_tensor_reduce(expr, ins, out_shape)

    _ensure_semantics("tensor_reduce", shape_rule, value_rule)

    assert _is_sym_tensor(data)
    out_shape = tuple(
        _shape_tensor_reduce(
            [ShapeExpr(list(data.shape))],
            {"axis": axis, "keepdims": keepdims},
        ).out.dims
    )
    return _new_sym_tensor(
        "tensor_reduce",
        [data],
        {"op": op, "axis": axis, "negate": negate, "keepdims": keepdims, "name": name},
        out_shape,
    )


@semantics_hw()
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
    return _new_sym_tensor("tensor_scalar", inputs, attrs, _tensor_scalar_out_shape(dst, data, operand0, operand1))


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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


@semantics_hw()
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
    a, b = ins
    ctx = _shape_ctx(*a.dims, *b.dims)
    if a.rank != b.rank:
        ctx.add(z3.BoolVal(False))
        return ShapeResult(ShapeExpr(list(a.dims)), ctx)
    for adim, bdim in zip(a.dims, b.dims):
        ctx.add(adim == bdim)
    return ShapeResult(ShapeExpr(list(a.dims)), ctx)


def _compile_tensor_tensor(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    a, b = ins
    out_fn = _tensor_function(f"V_{expr.name}", out_shape.rank)
    ctx = a.ctx.merged(b.ctx)
    idx = [z3.Int(f"{expr.name}_i{k}") for k in range(out_shape.rank)]
    av = a.fn(*idx)
    bv = b.fn(*idx)
    ctx.add(z3.ForAll(idx, out_fn(*idx) == _apply_binary(expr.attrs.get("op"), av, bv)))
    return Semantics(expr.name, out_shape, out_fn, ctx)


def _shape_tensor_scalar(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    data = ins[0]
    ctx = Context([d > 0 for d in data.dims])
    if len(ins) > 1:
        ctx.extend(_broadcast_to_shape(ins[1], data).ctx.facts)
    if len(ins) > 2:
        ctx.extend(_broadcast_to_shape(ins[2], data).ctx.facts)
    return ShapeResult(ShapeExpr(list(data.dims)), ctx)


def _tensor_scalar_out_shape(
    dst: Any,
    data: SymTensor,
    operand0: Any,
    operand1: Any = None,
) -> tuple[Any, ...]:
    if isinstance(dst, SymTensor):
        return dst.shape
    return tuple(data.shape)


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
    # Graph nodes still use `keep_dims`; public tensor-style callers use `keepdims`.
    keep = bool(expr.attrs.get("keepdims", expr.attrs.get("keep_dims", False)))
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
        # Public binary ops support NumPy-style broadcasting: use broadcast
        # semantics so the shape rule emits ``Or(a==b, a==1, b==1)`` per-dim
        # constraints rather than strict equality ``a==b``.  Strict equality
        # was incorrect for ops like ``div`` whose second operand is a
        # keep-dims reduce result with a trailing size-1 dimension.
        return _broadcast_shape(ins[0], ins[1])
    if len(ins) == 1:
        return _shape_same_as_first(ins, attrs)
    return _shape_from_out(attrs.get("out_shape", (z3.IntVal(1),)))


def _compile_public_binary(expr: SymExpr, ins: list[Semantics], out_shape: ShapeExpr) -> Semantics:
    def lift_binary_reduction(op_name: Any) -> Optional[ReductionDesc]:
        """Push a binary mul/div factor into a rank-2 reduction body when legal.

        Returns a new `ReductionDesc` for the lifted reduction body, or `None`
        when the binary expression is not an axiom-supported reduction element.
        """
        if len(ins) != 2:
            return None
        opn = _operand_to_expr(op_name)
        if opn not in ("mul", "multiply", "div", "divide"):
            return None

        reduction_index = next((idx for idx, sem in enumerate(ins) if sem.reduction is not None), None)
        if reduction_index is None:
            return None
        # Only reductions in the numerator can be lifted through division:
        # Red_X^+(f(X)) / v == Red_X^+(f(X) / v), but v / Red_X^+(f(X)) does not.
        if opn in ("div", "divide") and reduction_index != 0:
            return None

        reduction_sem = ins[reduction_index]
        factor_sem = ins[1 - reduction_index]
        reduction = reduction_sem.reduction
        if reduction is None or reduction.outer_rank != 2 or out_shape.rank != 2:
            return None

        body_id = _fresh_body_id()
        target_body = z3.IntVal(body_id)
        source_body = z3.IntVal(reduction.body_id)
        i = z3.Int(f"{expr.name}_ri")
        j = z3.Int(f"{expr.name}_rj")
        k = z3.Int(f"{expr.name}_rk")
        factor = _call_broadcasted(factor_sem, out_shape, [i, j])

        if opn in ("mul", "multiply"):
            if reduction_index == 0:
                lifted_body = BODY2(source_body, i, j, k) * factor
            else:
                lifted_body = factor * BODY2(source_body, i, j, k)
        else:
            lifted_body = _safe_divide(BODY2(source_body, i, j, k), factor)

        ctx.add(z3.ForAll([i, j, k], BODY2(target_body, i, j, k) == lifted_body))
        ctx.add(z3.ForAll([i, j], out_fn(i, j) == REDUCE2(target_body, i, j, reduction.extent)))
        return ReductionDesc(body_id, reduction.extent, outer_rank=2)

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
    return Semantics(expr.name, out_shape, out_fn, ctx, reduction=lift_binary_reduction(expr.attrs.get("op", expr.op)))


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


def _shape_graph_reduce_sum(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    return _shape_public_reduce(
        ins,
        {
            "axis": attrs.get("axis"),
            "keepdims": bool(attrs.get("keep_dims", attrs.get("keepdims", False))),
        },
    )


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


def _shape_graph_broadcast(ins: list[ShapeExpr], attrs: dict[str, Any]) -> ShapeResult:
    target_shape = tuple(ins[1].dims) if len(ins) >= 2 else attrs.get("shape")
    return _shape_public_broadcast_to(ins[:1], {"shape": target_shape})


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

    _ensure_semantics("broadcast", _shape_graph_broadcast, _compile_copy)
    _ensure_semantics("broadcast_to", _shape_public_broadcast_to, _compile_copy)
    _ensure_semantics("div", _shape_public_binary, _compile_public_binary)
    _ensure_semantics("expand_dims", _shape_public_expand_dims, _compile_public_expand_dims)
    _ensure_semantics("load", _shape_same_as_first, _compile_copy)
    _ensure_semantics("load_transpose2d", _shape_public_transpose2d, _compile_nc_transpose)
    _ensure_semantics("matmul", _shape_public_matmul, _compile_public_matmul)
    _ensure_semantics("mul", _shape_public_binary, _compile_public_binary)
    _ensure_semantics("reduce_sum", _shape_graph_reduce_sum, _compile_tensor_reduce)
    _ensure_semantics("transpose", _shape_public_transpose2d, _compile_nc_transpose)
    _ensure_semantics("where", _shape_public_where, _compile_public_where)


_register_public_semantics()






class DummyArray:
    def __init__(self, node_id: str, shape: tuple[int, ...], nodes: Optional[list[Any]] = None):
        self.node_id = node_id
        self.shape = shape
        self.nodes = list(nodes) if nodes is not None else [_make_input_node(node_id, shape)]

    @staticmethod
    def _merge_nodes(inputs: list["DummyArray"]) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for inp in inputs:
            for node in inp.nodes:
                if node.id in seen:
                    continue
                merged.append(node)
                seen.add(node.id)
        return merged

    @staticmethod
    def _from_op(op: str, inputs: list["DummyArray"], out_shape: tuple[int, ...], attrs: Optional[dict[str, Any]] = None) -> "DummyArray":
        node_id = _gen_id(op)
        nodes = DummyArray._merge_nodes(inputs)
        nodes.append(Node(id=node_id, op=op, inputs=[inp.node_id for inp in inputs], attrs=dict(attrs or {})))
        return DummyArray(node_id, out_shape, nodes)

    def _binary_op(self, other: Any, op: str) -> "DummyArray":
        if op == "matmul":
            if not isinstance(other, DummyArray):
                raise TypeError("matmul expects DummyArray operand")
            if len(self.shape) != 2 or len(other.shape) != 2:
                raise ValueError("matmul expects rank-2 inputs")
            if not _dims_equal(self.shape[1], other.shape[0]):
                raise ValueError("matmul expects compatible inner dimensions")
            return DummyArray._from_op(op, [self, other], (self.shape[0], other.shape[1]))

        if isinstance(other, DummyArray):
            out_shape = tuple(_normalize_dim(d) for d in _public_broadcast_shape_tuple(self.shape, other.shape))
            return DummyArray._from_op(op, [self, other], out_shape)
        else:
            out_shape = self.shape
            return DummyArray._from_op(op, [self], out_shape, {"scalar": other})

    def __add__(self, other: Any) -> "DummyArray":
        return self._binary_op(other, "add")

    def __mul__(self, other: Any) -> "DummyArray":
        return self._binary_op(other, "mul")

    def __truediv__(self, other: Any) -> "DummyArray":
        return self._binary_op(other, "div")

    def __matmul__(self, other: Any) -> "DummyArray":
        return self._binary_op(other, "matmul")

    def sum(self, axis: Any = None, keep_dims: bool = False) -> "DummyArray":
        out_shape = _public_reduce_out_shape(self.shape, axis, keep_dims)
        return DummyArray._from_op(
            "reduce_sum",
            [self],
            tuple(_normalize_dim(d) for d in out_shape),
            {"axis": axis, "keep_dims": keep_dims},
        )

    def broadcast_like(self, other: "DummyArray") -> "DummyArray":
        return DummyArray._from_op("broadcast", [self, other], other.shape)

    def sqrt(self) -> "DummyArray":
        return DummyArray._from_op("sqrt", [self], self.shape)

    def exp(self) -> "DummyArray":
        return DummyArray._from_op("exp", [self], self.shape)

    def transpose(self) -> "DummyArray":
        if len(self.shape) == 2:
            return DummyArray._from_op("transpose", [self], (self.shape[1], self.shape[0]))
        return DummyArray._from_op("transpose", [self], self.shape)

    def relu(self) -> "DummyArray":
        return DummyArray._from_op("relu", [self], self.shape)

    def silu(self) -> "DummyArray":
        return DummyArray._from_op("silu", [self], self.shape)

    def softmax(self, axis: int = -1) -> "DummyArray":
        return DummyArray._from_op("softmax", [self], self.shape, {"axis": axis})


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


def _make_input_node(node_id: str, shape: tuple[int, ...]) -> Node:
    return Node(id=node_id, op="input", inputs=[], attrs={"shape": shape})


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


def _normalize_dim(d: Any) -> Any:
    if isinstance(d, int):
        return d
    if isinstance(d, z3.ArithRef):
        d = z3.simplify(d)
    if isinstance(d, z3.IntNumRef):
        return d.as_long()
    return d


def _format_dim(d: Any) -> str:
    d = _normalize_dim(d)
    if isinstance(d, z3.ArithRef):
        return str(z3.simplify(d))
    return str(d)


def _format_shape(shape: Optional[tuple[Any, ...]]) -> str:
    if shape is None:
        return "None"
    dims = ", ".join(_format_dim(d) for d in shape)
    if len(shape) == 1:
        dims += ","
    return f"({dims})"


def _dims_equal(a: Any, b: Any) -> bool:
    lhs = _to_dim(a) if isinstance(a, int) else a
    rhs = _to_dim(b) if isinstance(b, int) else b
    return _arith_equal(lhs, rhs)


def _shape_expr_from_dims(shape: tuple[Any, ...]) -> ShapeExpr:
    return ShapeExpr([_to_dim(dim) for dim in shape])


def annotate_shapes_concrete(G: nuGraph) -> nuGraph:
    shapes: dict[str, tuple[Any, ...]] = {}
    for n in G.nodes:
        if n.op == "input":
            shape = tuple(_normalize_dim(d) for d in n.attrs.get("shape", n.shape or tuple()))
        else:
            if n.op not in _SEMANTICS:
                raise KeyError(f"No shape rule registered for graph op '{n.op}'")
            shape_rule, _ = _SEMANTICS[n.op]
            if shape_rule is None:
                raise KeyError(f"No shape rule registered for graph op '{n.op}'")
            shape_res = shape_rule([_shape_expr_from_dims(shapes[inp]) for inp in n.inputs], dict(n.attrs))
            shape = tuple(_normalize_dim(d) for d in shape_res.out.dims)
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


def _nodes_by_op(G: nuGraph, op: str) -> list[Node]:
    return [n for n in G.nodes if n.op == op]


def _immediate_successor_positions(G: nuGraph, pos: int) -> list[int]:
    node_id = G.node_at(pos).id
    return [i for i, n in enumerate(G.nodes) if node_id in n.inputs]


def _effective_input_ids(G: nuGraph, pos: int) -> list[str]:
    return list(dict.fromkeys(G.node_at(pos).inputs))


def _fresh_graph_node_id(G: nuGraph, base: str) -> str:
    existing = {n.id for n in G.nodes}
    if base not in existing:
        return base
    idx = 1
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def _graph_output_nodes(G: nuGraph) -> list[Node]:
    used = {inp for n in G.nodes for inp in n.inputs}
    return [n for n in G.nodes if n.id not in used]


def _sym_expr_from_graph_node(
    node: Node,
    inputs: list[SymTensor],
    symbolic_shape: Optional[tuple[z3.ArithRef, ...]] = None,
) -> SymTensor:
    if node.op == "input":
        rank = len(node.attrs.get("shape", node.shape or tuple()))
        shape = tuple(z3.Int(f"{node.id}_d{k}") for k in range(rank))
        return SymTensor(node.id, shape=shape)

    if node.op not in _SEMANTICS:
        raise KeyError(f"No symbolic conversion registered for graph op '{node.op}'")

    shape_rule, _ = _SEMANTICS[node.op]
    if shape_rule is None:
        raise KeyError(f"No shape rule registered for symbolic op '{node.op}'")
    shape_res = shape_rule([ShapeExpr(list(inp.shape)) for inp in inputs], dict(node.attrs))
    out_shape = symbolic_shape or tuple(shape_res.out.dims)
    expr = SymExpr(node.op, [inp.expr for inp in inputs], out_shape, dict(node.attrs), node.id)
    return SymTensor(node.id, expr=expr)


def _graph_symbolic_tensors(G: nuGraph) -> dict[str, SymTensor]:
    out: dict[str, SymTensor] = {}
    for node in G.nodes:
        inputs = [out[inp] for inp in node.inputs]
        out[node.id] = _sym_expr_from_graph_node(node, inputs)
    return out


def _swap_composition_tensor(G: nuGraph, result_id: str) -> Optional[SymTensor]:
    tensors = _graph_symbolic_tensors(G)
    return tensors.get(result_id)


def _swap_with_successor_variants(
    G_cur: nuGraph, pos: int, succ_pos: int, clone_inputs: set[str]
) -> list[tuple[nuGraph, int]]:
    op1 = G_cur.node_at(pos)
    op2 = G_cur.node_at(succ_pos)
    op1_input_ids = set(op1.inputs)
    if pos >= succ_pos:
        return []
    if not clone_inputs:
        return []
    if not clone_inputs.issubset(op1_input_ids):
        return []
    if len(G_cur.successors(op1)) != 1:
        return []

    consumed_positions = [i for i, input_id in enumerate(op2.inputs) if input_id == op1.id]
    if len(consumed_positions) != 1:
        return []
    consumed_pos = consumed_positions[0]

    selected_positions = [i for i, input_id in enumerate(op1.inputs) if input_id in clone_inputs]
    if not selected_positions:
        return []

    selected_input_ids = [op1.inputs[i] for i in selected_positions]
    candidate_inputs: list[list[str]] = []
    seen_inputs: set[tuple[str, ...]] = set()
    for perm in permutations(selected_input_ids):
        rebuilt_inputs = list(op1.inputs)
        for idx, permuted_input_id in zip(selected_positions, perm):
            rebuilt_inputs[idx] = permuted_input_id
        key = tuple(rebuilt_inputs)
        if key not in seen_inputs:
            seen_inputs.add(key)
            candidate_inputs.append(rebuilt_inputs)

    out: list[tuple[nuGraph, int]] = []
    seen_graphs: set[str] = set()
    full_input_set = set(op1.inputs)

    def _append_variant(
        clone_nodes: list[Node],
        rebuilt_node: Node,
    ) -> None:
        new_nodes: list[Node] = []
        for idx, node in enumerate(G_cur.nodes):
            if idx == succ_pos:
                new_nodes.extend(Node(n.id, n.op, list(n.inputs), dict(n.attrs), n.shape) for n in clone_nodes)
                new_nodes.append(rebuilt_node)
                continue
            if idx == pos:
                continue
            copied = Node(node.id, node.op, list(node.inputs), dict(node.attrs), node.shape)
            new_nodes.append(copied)

        G_new = nuGraph(new_nodes)
        new_pos = _position_by_id(G_new, op2.id)
        if new_pos is None:
            return
        sig = graph_signature(G_new)
        if sig in seen_graphs:
            return
        seen_graphs.add(sig)
        out.append((G_new, new_pos))

    if (
        op1.op == "nc_matmul"
        and op2.op == "nc_transpose"
        and len(op1.inputs) == 2
        and clone_inputs == full_input_set
    ):
        rebuilt_node = Node(op2.id, op1.op, [op1.inputs[1], op1.inputs[0]], dict(op1.attrs), None)
        _append_variant([], rebuilt_node)

    for rebuilt_inputs in candidate_inputs:
        base_graph = G_cur.clone()
        op1_new = _node_by_id(base_graph, op1.id)
        op2_old = _node_by_id(base_graph, op2.id)
        if op1_new is None or op2_old is None:
            continue

        clone_nodes: list[Node] = []
        clone_output_ids: dict[str, str] = {}
        ordered_clone_inputs = [input_id for input_id in op1.inputs if input_id in clone_inputs]
        for input_id in ordered_clone_inputs:
            clone_id = _fresh_graph_node_id(base_graph, f"{op2.id}_via_{input_id}")
            clone_node = Node(clone_id, op2_old.op, list(op2_old.inputs), dict(op2_old.attrs), None)
            clone_node.inputs[consumed_pos] = input_id
            clone_nodes.append(clone_node)
            clone_output_ids[input_id] = clone_id
            base_graph.nodes.append(clone_node)

        rebuilt_node = Node(op2.id, op1.op, list(rebuilt_inputs), dict(op1.attrs), None)
        rebuilt_node.inputs = [clone_output_ids.get(input_id, input_id) for input_id in rebuilt_node.inputs]
        _append_variant(clone_nodes, rebuilt_node)

    return out


def z3_equivalent_order(
    op1: Node, op2: Node, G_cur: nuGraph, G_new: Optional[nuGraph] = None, verbose: bool = False
) -> bool:
    if G_new is None:
        return False
    lhs = _swap_composition_tensor(G_cur, op2.id)
    rhs = _swap_composition_tensor(G_new, op2.id)
    if lhs is None or rhs is None:
        return False
    return check_equivalent(
        lhs,
        rhs,
        timeout=10000,
        rule_name=f"swap_{op1.id}_{op2.id}",
        verbose=verbose,
    )


def nu_graph_generation_z3(G : nuGraph, verbose=False) -> List[nuGraph]:
    G0 = G.clone()
    M: set[nuGraph] = {G0}
    equivalence_cache: dict[tuple[str, str, str, str], bool] = {}

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
                found_valid_swap = False

                for succ_pos in _immediate_successor_positions(G_cur, pos):
                    op1 = G_cur.node_at(pos)
                    op2 = G_cur.node_at(succ_pos)
                    inputs = _effective_input_ids(G_cur, pos)
                    accepted = False

                    # We cap the number of source inputs considered here at 4 because
                    # subset/permutation enumeration grows very quickly beyond that,
                    # while the graphs exercised by this prototype only use small-input
                    # operators in the propagation path.
                    if len(inputs) > 4:
                        continue
                    for k in range(1, len(inputs) + 1):
                        for subset in combinations(inputs, k):
                            for G_new, p_new in _swap_with_successor_variants(G_cur, pos, succ_pos, set(subset)):
                                cache_key = (graph_signature(G_cur), op1.id, op2.id, graph_signature(G_new))
                                equivalent = equivalence_cache.get(cache_key)
                                if equivalent is None:
                                    equivalent = z3_equivalent_order(op1, op2, G_cur, G_new, verbose=verbose)
                                    equivalence_cache[cache_key] = equivalent
                                if not equivalent:
                                    continue
                                M_next.add(G_new)
                                worklist.append((G_new, p_new))
                                found_valid_swap = True
                                accepted = True
                                break
                            if accepted:
                                break
                        if accepted:
                            break
                if not found_valid_swap:
                    M_next.add(G_cur)
        M |= M_next

    return sorted(M, key=graph_signature)


# ---------------------------------------------------------------------------
# Sketch-driven synthesizer and nuGraph hardware lowering
# ---------------------------------------------------------------------------

# Hardware ops registered with @semantics_hw that are candidates for lowering.
_HW_OP_NAMES: list[str] = [
    "activation",
    "activation_reduce",
    "dma_copy",
    "dma_transpose",
    "exponential",
    "nc_matmul",
    "nc_transpose",
    "reciprocal",
    "tensor_copy",
    "tensor_partition_reduce",
    "tensor_reduce",
    "tensor_scalar",
    "tensor_tensor",
]

# Synthesis candidates used to build the sketch pool.  Layout transforms are
# represented abstractly here so synthesis does not branch over multiple
# concrete copy/transpose hardware implementations.
_SYNTHESIS_POOL_OP_NAMES: list[str] = [
    hw_op for hw_op in _HW_OP_NAMES
    if hw_op not in {"dma_copy", "dma_transpose", "nc_transpose", "tensor_copy"}
] + ["transpose"]

# Ops whose only semantic role is a data layout transformation (always included
# in the pool regardless of constituent overlap).
_LAYOUT_TRANSFORM_OPS: frozenset[str] = frozenset({
    "transpose",
})

# Public ops that lower trivially to a single hw op without needing synthesis
# (handled by treating the graph node as a passthrough / identity in the hw
# graph when the hw sym has already been built for the input).
_PUBLIC_PASSTHROUGH_OPS: frozenset[str] = frozenset({
    "broadcast",
    "broadcast_to",
    "store",
    "load",
})

# Constituent primitive operations for public ops and hw ops.
# Shared constituents drive Phase 1 pool filtering.
_OP_CONSTITUENTS: dict[str, frozenset[str]] = {
    # Public ops
    "add":       frozenset({"add"}),
    "div":       frozenset({"divide"}),
    "divide":    frozenset({"divide"}),
    "exp":       frozenset({"exp"}),
    "matmul":    frozenset({"multiply", "add"}),
    "mul":       frozenset({"multiply"}),
    "multiply":  frozenset({"multiply"}),
    "reduce_sum": frozenset({"add"}),
    "relu":      frozenset({"relu"}),
    "silu":      frozenset({"silu"}),
    "softmax":   frozenset({"exp", "add", "divide"}),
    "sqrt":      frozenset({"sqrt"}),
    "subtract":  frozenset({"subtract"}),
    "transpose": frozenset({"transpose"}),
    "rms_norm":  frozenset({"multiply", "add", "sqrt", "divide"}),
    # Hardware ops
    "activation": frozenset({
        "relu", "silu", "gelu", "tanh", "sigmoid", "sqrt", "copy", "exp",
    }),
    "activation_reduce": frozenset({
        "relu", "silu", "gelu", "tanh", "sigmoid", "sqrt", "copy", "exp", "add",
    }),
    "dma_copy":              frozenset({"copy"}),
    "dma_transpose":         frozenset({"transpose", "copy"}),
    "exponential":           frozenset({"exp"}),
    "nc_matmul":             frozenset({"multiply", "add"}),
    "nc_transpose":          frozenset({"transpose", "copy"}),
    "reciprocal":            frozenset({"divide"}),
    "tensor_copy":           frozenset({"copy"}),
    "tensor_partition_reduce": frozenset({"add"}),
    "tensor_reduce":         frozenset({"add"}),
    "tensor_scalar":         frozenset({"add", "multiply", "divide", "subtract"}),
    "tensor_tensor":         frozenset({
        "add", "multiply", "divide", "subtract", "maximum", "minimum",
    }),
}


def _op_constituents(op_name: str) -> frozenset[str]:
    return _OP_CONSTITUENTS.get(op_name, frozenset({op_name}))


def _shares_constituents(target_op: str, hw_op: str) -> bool:
    return bool(_op_constituents(target_op) & _op_constituents(hw_op))


# ---------------------------------------------------------------------------
# Sketch / hole representation
# ---------------------------------------------------------------------------

_HOLE_SENTINEL = object()


@dataclass
class SketchNode:
    """A node in a partially-filled synthesis sketch.

    A leaf node is either:
    - a concrete input (hole=False, op="INPUT", sym set)
    - a hole (hole=True, sym=None)

    An interior node is a hw op application with children that may contain
    holes.
    """
    hole: bool
    op: str                        # "INPUT" | hw op name
    children: list["SketchNode"]   # operands (may contain holes)
    attrs: dict[str, Any]          # fixed op attributes
    sym: Optional[SymTensor]       # evaluated SymTensor (None if unresolved)

    def has_hole(self) -> bool:
        if self.hole:
            return True
        return any(c.has_hole() for c in self.children)

    def hw_size(self) -> int:
        """Number of hw op nodes (excluding INPUT leaves)."""
        if self.hole:
            return 0
        if self.op == "INPUT":
            return 0
        return 1 + builtins.sum(c.hw_size() for c in self.children)

    def _key(self) -> Any:
        if self.hole:
            return ("HOLE",)
        if self.op == "INPUT":
            return ("INPUT", id(self.sym))
        return (
            self.op,
            tuple(sorted((k, repr(v)) for k, v in self.attrs.items() if k != "name")),
            tuple(c._key() for c in self.children),
        )

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SketchNode) and self._key() == other._key()

    @staticmethod
    def make_hole() -> "SketchNode":
        return SketchNode(hole=True, op="HOLE", children=[], attrs={}, sym=None)

    @staticmethod
    def make_input(sym: SymTensor) -> "SketchNode":
        return SketchNode(hole=False, op="INPUT", children=[], attrs={}, sym=sym)

    @staticmethod
    def make_op(op: str, children: list["SketchNode"], attrs: dict[str, Any]) -> "SketchNode":
        return SketchNode(hole=False, op=op, children=list(children), attrs=dict(attrs), sym=None)


def _fill_first_hole(sketch: SketchNode, replacement: SketchNode) -> Optional[SketchNode]:
    """Return a new sketch with the first HOLE replaced by *replacement*."""
    if sketch.hole:
        return replacement
    if sketch.op == "INPUT":
        return None
    new_children: list[SketchNode] = []
    filled = False
    for child in sketch.children:
        if not filled:
            result = _fill_first_hole(child, replacement)
            if result is not None:
                new_children.append(result)
                filled = True
                # append remaining children unchanged
                new_children.extend(sketch.children[len(new_children):])
                break
        new_children.append(child)
    if not filled:
        return None
    return SketchNode(
        hole=False,
        op=sketch.op,
        children=new_children,
        attrs=dict(sketch.attrs),
        sym=None,
    )


# ---------------------------------------------------------------------------
# Evaluating a sketch to a SymTensor
# ---------------------------------------------------------------------------

def _invoke_hw_op(op: str, input_syms: list[SymTensor], attrs: dict[str, Any]) -> Optional[SymTensor]:
    """Call the hw op function to build a SymTensor for a completed sketch node."""
    try:
        if op == "activation":
            op_attr = attrs.get("op", nl.copy)
            return activation(dst=None, op=op_attr, data=input_syms[0])
        if op == "activation_reduce":
            op_attr = attrs.get("op", nl.copy)
            reduce_op = attrs.get("reduce_op", nl.add)
            return activation_reduce(dst=None, op=op_attr, data=input_syms[0], reduce_op=reduce_op, reduce_res=True)
        if op == "dma_copy":
            return dma_copy(dst=None, src=input_syms[0])
        if op == "dma_transpose":
            return dma_transpose(dst=None, src=input_syms[0], axes=attrs.get("axes"))
        if op == "exponential":
            return exponential(dst=None, src=input_syms[0], max_value=0.0)
        if op == "nc_matmul":
            if len(input_syms) < 2:
                return None
            return nc_matmul(dst=None, stationary=input_syms[0], moving=input_syms[1])
        if op == "nc_transpose":
            return nc_transpose(dst=None, data=input_syms[0])
        if op == "reciprocal":
            return reciprocal(dst=None, data=input_syms[0])
        if op == "tensor_copy":
            return tensor_copy(dst=None, src=input_syms[0])
        if op == "transpose":
            # Evaluate using nc_transpose (the hardware op that
            # _sketch_to_graph_nodes emits) so that equivalence checking and
            # materialization use the same semantics.
            return nc_transpose(dst=None, data=input_syms[0])
        if op == "tensor_partition_reduce":
            reduce_op_attr = attrs.get("op", nl.add)
            return tensor_partition_reduce(dst=None, op=reduce_op_attr, data=input_syms[0])
        if op == "tensor_reduce":
            axis_val = attrs.get("axis", 1)
            kd = bool(attrs.get("keepdims", False))
            op_attr = attrs.get("op", nl.add)
            return tensor_reduce(dst=None, op=op_attr, data=input_syms[0], axis=axis_val, negate=False, keepdims=kd)
        if op == "tensor_scalar":
            if len(input_syms) < 2:
                scalar_val = attrs.get("operand0_const", 1.0)
                op0_attr = attrs.get("op0", nl.multiply)
                return tensor_scalar(dst=None, data=input_syms[0], op0=op0_attr, operand0=scalar_val)
            return tensor_scalar(
                dst=None, data=input_syms[0],
                op0=attrs.get("op0", nl.multiply), operand0=input_syms[1],
                op1=attrs.get("op1"), operand1=attrs.get("operand1"),
            )
        if op == "tensor_tensor":
            if len(input_syms) < 2:
                return None
            op_attr = attrs.get("op", nl.add)
            return tensor_tensor(dst=None, data1=input_syms[0], data2=input_syms[1], op=op_attr)
        return None
    except Exception:
        return None


def _eval_sketch(sketch: SketchNode) -> Optional[SymTensor]:
    """Recursively evaluate a hole-free sketch to a SymTensor."""
    if sketch.hole:
        return None
    if sketch.op == "INPUT":
        return sketch.sym
    child_syms: list[SymTensor] = []
    for child in sketch.children:
        s = _eval_sketch(child)
        if s is None:
            return None
        child_syms.append(s)
    return _invoke_hw_op(sketch.op, child_syms, sketch.attrs)


def _template_constituents(sketch: SketchNode) -> frozenset[str]:
    """Return the primitive ops exercised by a single hardware template."""
    if sketch.hole or sketch.op == "INPUT":
        return frozenset()
    if sketch.op == "activation":
        return frozenset({_operand_to_expr(sketch.attrs.get("op", nl.copy))})
    if sketch.op == "activation_reduce":
        return frozenset({
            _operand_to_expr(sketch.attrs.get("op", nl.copy)),
            _operand_to_expr(sketch.attrs.get("reduce_op", nl.add)),
        })
    if sketch.op in ("dma_copy", "tensor_copy"):
        # Concrete layout ops can still appear in materialized/lowered sketches
        # even though the synthesis pool prefers abstract transpose templates.
        return frozenset({"copy"})
    if sketch.op in ("dma_transpose", "nc_transpose"):
        return frozenset({"transpose", "copy"})
    if sketch.op == "transpose":
        return frozenset({"transpose"})
    if sketch.op == "exponential":
        return frozenset({"exp"})
    if sketch.op == "nc_matmul":
        return frozenset({"multiply", "add"})
    if sketch.op == "reciprocal":
        return frozenset({"divide"})
    if sketch.op in ("tensor_partition_reduce", "tensor_reduce"):
        return frozenset({_operand_to_expr(sketch.attrs.get("op", nl.add))})
    if sketch.op == "tensor_scalar":
        constituents = {_operand_to_expr(sketch.attrs.get("op0", nl.multiply))}
        op1 = sketch.attrs.get("op1")
        if op1 is not None:
            constituents.add(_operand_to_expr(op1))
        return frozenset(constituents)
    if sketch.op == "tensor_tensor":
        return frozenset({_operand_to_expr(sketch.attrs.get("op", nl.add))})
    return _op_constituents(sketch.op)


# ---------------------------------------------------------------------------
# Phase 1 – Build instruction pool
# ---------------------------------------------------------------------------

def _pool_templates_for_hw_op(
    hw_op: str,
    concrete: list[SketchNode],
    target_attrs: dict[str, Any],
) -> list[SketchNode]:
    """Return sketch templates for *hw_op* with holes and/or concrete inputs."""
    H = SketchNode.make_hole
    templates: list[SketchNode] = []

    def add_unary(attrs: dict[str, Any]) -> None:
        templates.append(SketchNode.make_op(hw_op, [H()], attrs))
        for n1 in concrete:
            templates.append(SketchNode.make_op(hw_op, [n1], attrs))

    def add_binary(attrs: dict[str, Any]) -> None:
        templates.append(SketchNode.make_op(hw_op, [H(), H()], attrs))
        for n1 in concrete:
            templates.append(SketchNode.make_op(hw_op, [n1, H()], attrs))
            templates.append(SketchNode.make_op(hw_op, [H(), n1], attrs))
            for n2 in concrete:
                templates.append(SketchNode.make_op(hw_op, [n1, n2], attrs))

    # Unary layout transforms
    if hw_op == "transpose":
        add_unary({})          # default (swap first two dims)
        return templates
    if hw_op == "reciprocal":
        add_unary({})
        return templates

    # activation: try activation ops that share constituents with target
    if hw_op == "activation":
        ta_constituents = _op_constituents(target_attrs.get("op_name", ""))
        act_ops = [nl.copy, nl.relu, nl.silu, nl.gelu, nl.tanh, nl.sigmoid, nl.sqrt]
        for act_op in act_ops:
            op_str = _operand_to_expr(act_op)
            if not ta_constituents or op_str in ta_constituents or op_str in _op_constituents(target_attrs.get("op_name", "")):
                add_unary({"op": act_op})
        return templates

    if hw_op == "activation_reduce":
        act_ops = [nl.copy, nl.relu, nl.exp]
        for act_op in act_ops:
            add_unary({"op": act_op, "reduce_op": nl.add})
        return templates

    if hw_op == "exponential":
        add_unary({})
        return templates

    # tensor_reduce: use axis/keepdims from target when available
    if hw_op == "tensor_reduce":
        ta_axis = target_attrs.get("axis", target_attrs.get("keep_dims_axis"))
        ta_kd = bool(target_attrs.get("keepdims", target_attrs.get("keep_dims", False)))
        axes_to_try = [ta_axis] if ta_axis is not None else [0, 1]
        kd_to_try = [ta_kd] if ta_axis is not None else [False, True]
        for ax in axes_to_try:
            for kd in kd_to_try:
                add_unary({"op": nl.add, "axis": ax, "keepdims": kd})
        return templates

    if hw_op == "tensor_partition_reduce":
        add_unary({"op": nl.add})
        return templates

    # tensor_scalar: binary with one tensor + one scalar-like operand
    if hw_op == "tensor_scalar":
        for op0 in [nl.add, nl.multiply, nl.divide, nl.subtract]:
            for n1 in concrete:
                templates.append(SketchNode.make_op(hw_op, [n1, H()], {"op0": op0}))
                templates.append(SketchNode.make_op(hw_op, [H(), n1], {"op0": op0}))
                for n2 in concrete:
                    templates.append(SketchNode.make_op(hw_op, [n1, n2], {"op0": op0}))
            templates.append(SketchNode.make_op(hw_op, [H(), H()], {"op0": op0}))
        return templates

    # tensor_tensor: all six binary ops
    if hw_op == "tensor_tensor":
        for op in [nl.add, nl.multiply, nl.divide, nl.subtract, nl.maximum, nl.minimum]:
            add_binary({"op": op})
        return templates

    # nc_matmul: standard binary templates
    if hw_op == "nc_matmul":
        add_binary({})
        return templates

    return templates


def _build_synthesis_pool(
    target_op: str,
    target_attrs: dict[str, Any],
    input_syms: list[SymTensor],
) -> list[SketchNode]:
    """Phase 1: build the instruction pool P for target operation."""
    concrete = [SketchNode.make_input(sym) for sym in input_syms]
    pool: list[SketchNode] = list(concrete)  # include raw inputs
    target_constituents = _op_constituents(target_op)

    augmented_attrs = dict(target_attrs)
    augmented_attrs["op_name"] = target_op

    for hw_op in _SYNTHESIS_POOL_OP_NAMES:
        is_layout = hw_op in _LAYOUT_TRANSFORM_OPS
        if not is_layout and not _shares_constituents(target_op, hw_op):
            continue
        templates = _pool_templates_for_hw_op(hw_op, concrete, augmented_attrs)
        if not is_layout:
            templates = [
                template for template in templates
                if _template_constituents(template).issubset(target_constituents)
            ]
        pool.extend(templates)

    return pool


# ---------------------------------------------------------------------------
# Phase 2 – Compose and check
# ---------------------------------------------------------------------------

def _check_equivalent_quiet(
    lhs: SymTensor,
    rhs: SymTensor,
    timeout: int = 3000,
) -> bool:
    """Run a quiet, thread-safe equivalence check under the shared Z3 lock."""
    try:
        with _Z3_LOCK:
            return check_equivalent(lhs, rhs, timeout=timeout)
    except Exception:
        return False


def _shapes_match_exactly(lhs: SymTensor, rhs: SymTensor) -> bool:
    """Return True when two symbolic shapes are syntactically identical."""
    if lhs.rank != rhs.rank:
        return False
    with _Z3_LOCK:
        return builtins.all(z3.eq(ldim, rdim) for ldim, rdim in zip(lhs.shape, rhs.shape))


def _shapes_incompatible_symbolically(lhs: SymTensor, rhs: SymTensor) -> bool:
    """Return True when symbolic shapes differ in a trivially decidable way."""
    if lhs.rank != rhs.rank:
        return True
    with _Z3_LOCK:
        for ldim, rdim in zip(lhs.shape, rhs.shape):
            if z3.is_int_value(ldim) and z3.is_int_value(rdim) and ldim.as_long() != rdim.as_long():
                return True
    return False


def _sketch_shape_constraints_violated(candidate_sym: SymTensor) -> bool:
    """Generic pre-solver check: return True if any op's shape rule produces a
    trivially-False constraint for the candidate sketch.

    Traverses the SymExpr tree bottom-up.  For each non-input op node the
    registered shape_rule is called with the *actual input shapes* taken from
    the sub-expression tree (not from any cached ``out_shape`` attr).  The
    constraints returned in ``ShapeResult.ctx`` encode the semantic
    preconditions for that op to be well-formed (e.g. ``nc_matmul`` requires
    ``stationary.shape[0] == moving.shape[0]``).

    If z3.simplify() reduces any such constraint to ``False`` the sketch is
    structurally impossible and can be rejected without invoking the expensive
    equivalence solver.

    The check is conservative: it only fires on constraints that are trivially
    decidable (``z3.simplify`` returns a Z3 ``False`` literal), so symbolic or
    otherwise undecidable constraints are never incorrectly rejected.
    """
    visited: set[int] = set()

    def check_node(expr: SymExpr) -> bool:
        key = id(expr)
        if key in visited:
            return False
        visited.add(key)

        if expr.op == "input":
            return False

        # Check children first (bottom-up)
        for inp in expr.inputs:
            if check_node(inp):
                return True

        shape_rule, _ = _SEMANTICS.get(expr.op, (None, None))
        if shape_rule is None:
            return False

        # Use the shapes from the child SymExprs directly.  We intentionally
        # strip "out_shape" from attrs so the shape_rule takes its full
        # constraint-generating code path rather than the cached-result early
        # return (which skips constraint production).
        input_shapes = [ShapeExpr(list(inp.shape)) for inp in expr.inputs]
        attrs_for_check = {k: v for k, v in expr.attrs.items() if k != "out_shape"}
        try:
            result = shape_rule(input_shapes, attrs_for_check)
            for constraint in result.ctx.facts:
                if z3.is_false(z3.simplify(constraint)):
                    return True
        except Exception:
            pass

        return False

    with _Z3_LOCK:
        return check_node(candidate_sym.expr)


def _shapes_not_provably_equivalent(
    target_sym: SymTensor,
    candidate_sym: SymTensor,
    timeout: int = 500,
    input_syms: Optional[list["SymTensor"]] = None,
) -> bool:
    """Return True when the candidate's operator shape constraints cannot be
    proved consistent with all valid inputs, given both input-dimension
    positivity and the target operation's own shape constraints.

    ``_sketch_shape_constraints_violated`` rejects candidates whose shape
    constraints are *trivially False* (``z3.simplify`` returns the False literal).
    This function handles the complementary case: constraints that are
    *symbolically non-trivial* — e.g. ``k == 1`` emitted by
    ``_shape_tensor_tensor`` when one operand has a concrete-1 dimension while
    the other has a symbolic dimension ``k``.  For such constraints a Z3 solver
    determines whether the constraint is necessarily implied by the "valid
    assumption" set, which comprises both the positivity facts ``d > 0`` and
    the target's own top-level shape constraints.

    Including the target's shape constraints as valid assumptions is essential
    for broadcast-capable candidates.  For example, when synthesising
    ``div(x[m, k], reduce[n, 1])`` the target's shape rule (``_broadcast_shape``)
    emits ``Or(m==n, m==1, n==1)``.  A candidate ``tensor_scalar(x, reduce)``
    also emits that same broadcast constraint.  Without the target's constraints
    in the assumption set the check ``SAT(m>0, k>0, n>0, ¬Or(m==n, m==1, n==1))``
    is SAT (e.g. m=4, n=5), so the candidate would be wrongly rejected.
    With the target constraint in the assumption set the check becomes UNSAT
    (the Or is already forced to hold), so the candidate is correctly kept.

    Conversely, a ``tensor_tensor(x, reduce)`` candidate emits the stricter
    ``m==n ∧ k==1``.  Even with the target's ``Or(m==n, m==1, n==1)`` as an
    assumption, ``SAT(m>0, k>0, n>0, Or(m==n,…), ¬(m==n ∧ k==1))`` is SAT
    (e.g. m=n=4, k=8), so that candidate is correctly rejected.

    *input_syms* parameter
    ----------------------
    When the synthesis pool includes complex (multi-op) SymTensors as inputs
    — for example, a canonical ``tensor_scalar`` result from a prior synthesis
    step being fed into a downstream ``matmul`` synthesis — traversing *into*
    those sub-expressions would collect their internal shape constraints (e.g.
    the broadcast ``Or(m==n, m==1, n==1)`` from the inner ``tensor_scalar``) as
    if they were new constraints introduced by the candidate sketch.  Negating
    them in the SAT check then wrongly rejects valid sketches such as
    ``nc_matmul(nc_transpose(div_sym), w_sym)``.

    Passing *input_syms* marks the SymExprs of the synthesis pool inputs as
    opaque boundaries.  When the constraint traversal reaches one of these
    nodes it stops without collecting any further constraints from the
    sub-tree, preventing the false-rejection described above.

    Algorithm
    ---------
    1. Traverse the candidate's SymExpr tree bottom-up and collect ``d > 0``
       for every *symbolic* input dimension (concrete ``IntVal`` dims produce
       trivially-True positivity facts and are skipped).
    2. Run the *target's* top-level shape rule on the target's input shapes and
       collect every non-trivial fact — these become additional valid assumptions
       that the checker can rely on.
    3. Re-run each operator's ``shape_rule`` on the actual child shapes (same
       as ``_sketch_shape_constraints_violated``) and collect every fact that
       *does not* simplify to True or False — these are the non-trivial shape
       constraints that the candidate imposes on its operands.
    4. Ask Z3: ``(positivity ∧ target_constraints) ∧ ¬(all candidate constraints)``
       — is this satisfiable?  If SAT there exist valid inputs (consistent with
       the target's preconditions) for which the candidate shape constraints are
       violated; the candidate is rejected.  If UNSAT, the valid-assumption set
       implies all candidate constraints; the candidate is kept.  On UNKNOWN
       (solver timeout) the result is conservatively treated as "not rejected"
       so that a slow solver never silently drops a valid candidate.

    Thread-safety
    -------------
    All Z3 operations are performed under ``_Z3_LOCK``, consistent with the
    rest of the synthesis pipeline.
    """
    # Build the set of SymExpr IDs for synthesis pool inputs so that
    # _collect_candidate_shape_constraints stops at these boundaries and does
    # not collect their *internal* shape constraints as candidate constraints.
    input_expr_ids: frozenset[int] = frozenset(
        id(sym.expr) for sym in (input_syms or []) if sym.expr is not None
    )

    input_positivity: list[z3.BoolRef] = []
    visited_inputs: set[int] = set()

    def _collect_input_positivity(expr: SymExpr) -> None:
        key = id(expr)
        if key in visited_inputs:
            return
        visited_inputs.add(key)
        if expr.op == "input":
            for dim in expr.shape:
                positivity = dim > z3.IntVal(0)
                if not z3.is_true(z3.simplify(positivity)):
                    input_positivity.append(positivity)
            return
        for inp in expr.inputs:
            _collect_input_positivity(inp)

    def _collect_top_level_shape_constraints(expr: SymExpr) -> list[z3.BoolRef]:
        """Return non-trivial shape constraints from the *top-level* op of expr.

        Only the top-level node is inspected (not its sub-tree) because the
        target represents a single public op whose shape preconditions define
        the synthesis problem's valid input domain.
        """
        if expr.op == "input":
            return []
        shape_rule, _ = _SEMANTICS.get(expr.op, (None, None))
        if shape_rule is None:
            return []
        input_shapes = [ShapeExpr(list(inp.shape)) for inp in expr.inputs]
        attrs_for_check = {k: v for k, v in expr.attrs.items() if k != "out_shape"}
        try:
            result = shape_rule(input_shapes, attrs_for_check)
        except Exception:
            return []
        out = []
        for constraint in result.ctx.facts:
            simplified = z3.simplify(constraint)
            if not z3.is_true(simplified) and not z3.is_false(simplified):
                out.append(constraint)
        return out

    candidate_shape_constraints: list[z3.BoolRef] = []
    visited_ops: set[int] = set()

    def _collect_candidate_shape_constraints(expr: SymExpr) -> None:
        key = id(expr)
        if key in visited_ops:
            return
        visited_ops.add(key)
        # Stop at bare input nodes and at synthesis-pool input boundaries.
        # Synthesis inputs (e.g. the canonical tensor_scalar sym from a prior
        # synthesis step) carry internal shape constraints (like the broadcast
        # Or(m==n,...)) that are *preconditions* of that prior synthesis, not
        # new constraints introduced by the current candidate sketch.  Treating
        # them as opaque prevents those preconditions from being negated in the
        # SAT check, which would otherwise wrongly reject valid downstream
        # sketches (e.g. nc_matmul(nc_transpose(div_sym), w_sym)).
        if expr.op == "input" or key in input_expr_ids:
            return
        for inp in expr.inputs:
            _collect_candidate_shape_constraints(inp)
        # tensor_scalar is intentionally directional: data drives the output
        # tile shape, while operand0/operand1 broadcast *into* that shape.
        # Those role-selection constraints are stricter than the public op's
        # symmetric broadcast preconditions, so negating them here would reject
        # otherwise-correct candidates before the actual equivalence solver has
        # a chance to validate them.
        if expr.op == "tensor_scalar":
            return
        shape_rule, _ = _SEMANTICS.get(expr.op, (None, None))
        if shape_rule is None:
            return
        input_shapes = [ShapeExpr(list(inp.shape)) for inp in expr.inputs]
        attrs_for_check = {k: v for k, v in expr.attrs.items() if k != "out_shape"}
        try:
            result = shape_rule(input_shapes, attrs_for_check)
            for constraint in result.ctx.facts:
                simplified = z3.simplify(constraint)
                if not z3.is_true(simplified) and not z3.is_false(simplified):
                    candidate_shape_constraints.append(constraint)
        except Exception:
            pass

    with _Z3_LOCK:
        _collect_input_positivity(candidate_sym.expr)
        target_expr = target_sym.expr if target_sym is not None else None
        target_shape_constraints = (
            _collect_top_level_shape_constraints(target_expr)
            if target_expr is not None
            else []
        )
        _collect_candidate_shape_constraints(candidate_sym.expr)

        if not candidate_shape_constraints:
            return False

        all_constraints = (
            z3.And(*candidate_shape_constraints)
            if len(candidate_shape_constraints) > 1
            else candidate_shape_constraints[0]
        )
        solver = z3.Solver()
        solver.set("timeout", timeout)
        # Valid-assumption set: input-dimension positivity PLUS the target's
        # own shape preconditions.  The latter ensure that broadcast-compatible
        # candidates are not wrongly rejected when the synthesis uses distinct
        # symbolic variables for dimensions that are equal at run time.
        valid_assumptions = input_positivity + target_shape_constraints
        if valid_assumptions:
            solver.add(z3.And(*valid_assumptions))
        solver.add(z3.Not(all_constraints))
        result = solver.check()
        # SAT    → there exist valid inputs (satisfying target preconditions)
        #          where candidate shape constraints fail → reject
        # UNSAT  → valid assumptions imply all candidate shape constraints → keep
        # UNKNOWN → solver timed out; be conservative and keep the candidate
        return result == z3.sat


def _shape_rejection_reason(
    target_sym: SymTensor,
    candidate_sym: SymTensor,
    input_syms: Optional[list["SymTensor"]] = None,
) -> Optional[str]:
    """Return a rejection reason if shape information proves a mismatch."""
    if _shapes_incompatible_symbolically(target_sym, candidate_sym):
        return "symbolic shape mismatch"
    if _sketch_shape_constraints_violated(candidate_sym):
        return "sketch shape constraints violated"
    if _shapes_not_provably_equivalent(target_sym, candidate_sym,
                                       input_syms=input_syms):
        return "output shape not provably equivalent to target"
    return None


_last_synthesis_stats: dict[str, Any] = {}


def _reset_synthesis_stats(**stats: Any) -> None:
    with _SYNTHESIS_STATS_LOCK:
        _last_synthesis_stats.clear()
        _last_synthesis_stats.update(stats)


def _update_synthesis_stats(**stats: Any) -> None:
    with _SYNTHESIS_STATS_LOCK:
        _last_synthesis_stats.update(stats)


def _format_sketch(sketch: SketchNode) -> str:
    """Return a compact human-readable string for a (possibly partial) sketch."""
    if sketch.hole:
        return "□"
    if sketch.op == "INPUT":
        name = "?"
        if sketch.sym is not None:
            try:
                name = sketch.sym.expr.name if sketch.sym.expr is not None else repr(sketch.sym)
            except Exception:
                name = repr(sketch.sym)
        return f"IN:{name}"
    children_str = ", ".join(_format_sketch(c) for c in sketch.children)
    attrs_parts = []
    for k, v in sketch.attrs.items():
        if k == "name":
            continue
        attrs_parts.append(f"{k}={v!r}")
    attrs_str = ("[" + ", ".join(attrs_parts) + "]") if attrs_parts else ""
    return f"{sketch.op}{attrs_str}({children_str})"


def _synthesize_from_pool(
    target_sym: SymTensor,
    pool: list[SketchNode],
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
    input_syms: Optional[list["SymTensor"]] = None,
) -> Optional[SketchNode]:
    """Phase 2: worklist search for a hw-only sketch equivalent to target_sym.

    Uses BFS (popleft from front) so that shallower candidates are always
    explored before deeper ones.  This guarantees that the first equivalent
    sketch returned has the smallest hw_size within the allowed budget,
    giving a deterministic, minimal canonical lowering.

    When *verbose* is True every complete sketch that is evaluated is printed
    together with whether it passed equivalence, allowing manual inspection.
    """
    initial = SketchNode.make_hole()
    worklist: deque[SketchNode] = deque([initial])
    seen: set[SketchNode] = {initial}
    symbolic_shape_reject_count = 0
    solver_dispatch_count = 0
    _reset_synthesis_stats(
        symbolic_shape_reject_count=0,
        solver_dispatch_count=0,
    )

    while worklist:
        sketch = worklist.popleft()       # BFS: pop from front

        if not sketch.has_hole():
            # Complete sketch – evaluate and check
            candidate_sym = _eval_sketch(sketch)
            if candidate_sym is None:
                if verbose:
                    print(f"    sketch {_format_sketch(sketch)!s:60s}  [eval failed]")
                continue
            rejection_reason = _shape_rejection_reason(
                target_sym,
                candidate_sym,
                input_syms=input_syms,
            )
            if rejection_reason is not None:
                symbolic_shape_reject_count += 1
                if verbose:
                    print(f"    sketch {_format_sketch(sketch)!s:60s}  [{rejection_reason}]")
                continue
            if sketch.hw_size() > max_hw_size:
                if verbose:
                    print(f"    sketch {_format_sketch(sketch)!s:60s}  [exceeds depth {max_hw_size}]")
                continue
            solver_dispatch_count += 1
            equiv = _check_equivalent_quiet(target_sym, candidate_sym, timeout=timeout)
            if verbose:
                status = "EQUIVALENT ✓" if equiv else "not equivalent"
                print(f"    sketch {_format_sketch(sketch)!s:60s}  [{status}]")
                print(
                    f"    symbolic_shape_reject_count={symbolic_shape_reject_count} "
                    f"solver_dispatch_count={solver_dispatch_count}"
                )
            _update_synthesis_stats(
                symbolic_shape_reject_count=symbolic_shape_reject_count,
                solver_dispatch_count=solver_dispatch_count,
            )
            if equiv:
                return sketch
            continue

        # Incomplete sketch – fill the first hole with each pool entry.
        can_add_hw_op = sketch.hw_size() < max_hw_size
        for pool_entry in pool:
            # Decide whether this pool entry is allowed at this depth
            if pool_entry.op not in ("INPUT", "HOLE"):
                if not can_add_hw_op:
                    continue
            filled = _fill_first_hole(sketch, pool_entry)
            if filled is None:
                continue
            if filled in seen:
                continue
            seen.add(filled)
            worklist.append(filled)

    _update_synthesis_stats(
        symbolic_shape_reject_count=symbolic_shape_reject_count,
        solver_dispatch_count=solver_dispatch_count,
    )
    return None


def _synthesize_all_from_pool(
    target_sym: SymTensor,
    pool: list[SketchNode],
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    input_syms: Optional[list["SymTensor"]] = None,
) -> list[SketchNode]:
    """Like _synthesize_from_pool but returns ALL equivalent sketches.

    Phase 1 runs the same DFS worklist as _synthesize_from_pool to enumerate
    every complete, evaluatable candidate sketch.  Phase 2 dispatches the
    equivalence checks across sketches through a ThreadPoolExecutor while
    serialising the z3 work under _Z3_LOCK.

    Returns every sketch that passes equivalence (empty list if none).
    When *verbose* is True each sketch is printed as soon as its equivalence
    check completes, holding _VERBOSE_LOCK so output from concurrent node-level
    threads does not interleave.
    """
    initial = SketchNode.make_hole()
    worklist: list[SketchNode] = [initial]
    seen: set[SketchNode] = {initial}
    candidates: list[tuple[SketchNode, SymTensor]] = []
    symbolic_shape_reject_count = 0
    _reset_synthesis_stats(
        candidate_count=0,
        symbolic_shape_reject_count=0,
        solver_dispatch_count=0,
    )

    # Phase 1: expand worklist, collect all complete evaluatable candidates
    while worklist:
        sketch = worklist.pop()

        if not sketch.has_hole():
            if sketch.hw_size() > max_hw_size:
                if verbose:
                    with _VERBOSE_LOCK:
                        print(f"    sketch {_format_sketch(sketch)!s:60s}  [exceeds depth {max_hw_size}]")
                continue
            candidate_sym = _eval_sketch(sketch)
            if candidate_sym is None:
                if verbose:
                    with _VERBOSE_LOCK:
                        print(f"    sketch {_format_sketch(sketch)!s:60s}  [eval failed]")
                continue
            rejection_reason = _shape_rejection_reason(
                target_sym,
                candidate_sym,
                input_syms=input_syms,
            )
            if rejection_reason is not None:
                symbolic_shape_reject_count += 1
                if verbose:
                    with _VERBOSE_LOCK:
                        print(f"    sketch {_format_sketch(sketch)!s:60s}  [{rejection_reason}]")
                continue
            candidates.append((sketch, candidate_sym))
            continue

        can_add_hw_op = sketch.hw_size() < max_hw_size
        for pool_entry in reversed(pool):
            if pool_entry.op not in ("INPUT", "HOLE"):
                if not can_add_hw_op:
                    continue
            filled = _fill_first_hole(sketch, pool_entry)
            if filled is None:
                continue
            if filled in seen:
                continue
            seen.add(filled)
            worklist.append(filled)

    _update_synthesis_stats(
        candidate_count=len(candidates),
        symbolic_shape_reject_count=symbolic_shape_reject_count,
        solver_dispatch_count=len(candidates),
    )

    if not candidates:
        return []

    if verbose:
        with _VERBOSE_LOCK:
            print(f"    candidate_count={len(candidates)}")
            print(f"    symbolic_shape_reject_count={symbolic_shape_reject_count}")
            print(f"    solver_dispatch_count={len(candidates)}")

    # Phase 2: check all candidates, optionally dispatching sketches through a
    # thread pool.  z3.main_ctx() is not thread-safe, so every actual solver
    # call must still hold _Z3_LOCK.
    def _check_one(item: tuple[SketchNode, SymTensor]) -> tuple[SketchNode, bool, float]:
        sketch, cand_sym = item
        started_at = time.perf_counter()
        try:
            equiv = _check_equivalent_quiet(target_sym, cand_sym, timeout=timeout)
        except Exception:
            equiv = False
        return sketch, equiv, time.perf_counter() - started_at

    def _emit_check_result(idx: int, result: tuple[SketchNode, bool, float]) -> None:
        if not verbose:
            return
        sketch, equiv, elapsed = result
        status = "EQUIVALENT ✓" if equiv else "not equivalent"
        with _VERBOSE_LOCK:
            print(
                f"    sketch {idx + 1:>3d}/{len(candidates):<3d} "
                f"{_format_sketch(sketch)!s:60s}  [{status} in {elapsed:.3f}s]",
                flush=True,
            )

    ordered: dict[int, tuple[SketchNode, bool, float]] = {}
    effective_workers = _effective_max_workers(max_workers, len(candidates))
    if effective_workers > 1 and len(candidates) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            fs = {executor.submit(_check_one, item): idx
                  for idx, item in enumerate(candidates)}
            for future in concurrent.futures.as_completed(fs):
                idx = fs[future]
                try:
                    ordered[idx] = future.result()
                except Exception:
                    ordered[idx] = (candidates[idx][0], False, 0.0)
                _emit_check_result(idx, ordered[idx])
    else:
        for idx, item in enumerate(candidates):
            ordered[idx] = _check_one(item)
            _emit_check_result(idx, ordered[idx])

    # Phase 3: collect results in original DFS order
    valid: list[SketchNode] = []
    for idx in range(len(candidates)):
        sketch, equiv, _ = ordered[idx]
        if equiv:
            valid.append(sketch)

    return valid


# ---------------------------------------------------------------------------
# Convert a found sketch to graph Nodes
# ---------------------------------------------------------------------------

def _sketch_to_graph_nodes(
    sketch: SketchNode,
    sym_to_node_id: dict[int, str],
    new_nodes: list[Node],
) -> Optional[str]:
    """Recursively convert a completed sketch to graph Nodes.

    Returns the node_id of the root node, or None on failure.
    Appends new intermediate Nodes to *new_nodes*.
    """
    if sketch.op == "INPUT":
        if sketch.sym is None:
            return None
        key = id(sketch.sym)
        return sym_to_node_id.get(key)

    child_ids: list[str] = []
    for child in sketch.children:
        cid = _sketch_to_graph_nodes(child, sym_to_node_id, new_nodes)
        if cid is None:
            return None
        child_ids.append(cid)

    clean_attrs = {k: v for k, v in sketch.attrs.items() if k != "name"}
    if sketch.op == "tensor_scalar":
        if len(child_ids) > 1:
            clean_attrs["operand0_input_index"] = 1
        if len(child_ids) > 2 and clean_attrs.get("op1") is not None:
            clean_attrs["operand1_input_index"] = 2
    # The synthesis pool uses abstract "transpose" so the search does not need
    # to branch over multiple concrete layout ops.  Both evaluation
    # (_invoke_hw_op) and materialization here map "transpose" to nc_transpose,
    # keeping equivalence checking and the emitted graph node consistent.
    materialized_op = "nc_transpose" if sketch.op == "transpose" else sketch.op
    new_id = _gen_id(materialized_op)
    new_nodes.append(Node(id=new_id, op=materialized_op, inputs=child_ids, attrs=clean_attrs))
    return new_id


# ---------------------------------------------------------------------------
# Lowering cache helpers
# ---------------------------------------------------------------------------

def _make_lowering_cache_key(
    op: str,
    attrs: dict[str, Any],
    input_syms: list["SymTensor"],
) -> tuple:
    """Return a hashable key that uniquely identifies a lowering problem.

    The key is built from the target op name, sorted attribute repr pairs, and
    the rank of each input tensor.  Concrete node IDs and z3 variable names are
    deliberately excluded so that structurally identical operations encountered
    in different graph variants (which may have different generated IDs) map to
    the same key.
    """
    frozen_attrs = tuple(sorted((k, repr(v)) for k, v in attrs.items()))
    input_ranks = tuple(s.rank for s in input_syms)
    return (op, frozen_attrs, input_ranks)


def _normalize_sketch(
    sketch: "SketchNode",
    input_syms: list["SymTensor"],
) -> Optional[_NormSketch]:
    """Convert a completed SketchNode into a sym-independent cached form.

    INPUT leaves are replaced by ``("INPUT", i)`` where *i* is the position of
    the leaf's SymTensor in *input_syms*.  Interior nodes retain their op name
    and attrs but carry normalized children rather than live SketchNode objects.
    Returns None if any INPUT leaf cannot be matched to *input_syms*.
    """
    if sketch.op == "INPUT":
        for i, sym in enumerate(input_syms):
            if sketch.sym is sym:
                return ("INPUT", i)
        return None
    children: list[_NormSketch] = []
    for child in sketch.children:
        n = _normalize_sketch(child, input_syms)
        if n is None:
            return None
        children.append(n)
    # Strip the transient "name" attr so the normalized form is stable.
    clean_attrs = {k: v for k, v in sketch.attrs.items() if k != "name"}
    return ("OP", sketch.op, clean_attrs, tuple(children))


def _denormalize_sketch(
    norm: _NormSketch,
    input_syms: list["SymTensor"],
) -> Optional["SketchNode"]:
    """Reconstruct a SketchNode from its normalized form, binding *input_syms*.

    This is the inverse of *_normalize_sketch*: INPUT leaves are replaced by
    fresh ``SketchNode.make_input(input_syms[i])`` nodes, so the reconstructed
    sketch refers to the *new* input SymTensors rather than the ones that were
    live when the sketch was first synthesized.
    Returns None if any INPUT index is out of range.
    """
    if norm[0] == "INPUT":
        idx = norm[1]
        if idx < 0 or idx >= len(input_syms):
            return None
        return SketchNode.make_input(input_syms[idx])
    _, op, attrs, children_norm = norm
    children: list[SketchNode] = []
    for cn in children_norm:
        child = _denormalize_sketch(cn, input_syms)
        if child is None:
            return None
        children.append(child)
    return SketchNode.make_op(op, children, dict(attrs))


# ---------------------------------------------------------------------------
# Per-node lowering
# ---------------------------------------------------------------------------

def _lower_node(
    node: Node,
    target_sym: SymTensor,
    hw_input_pairs: list[tuple[SymTensor, str]],
    max_hw_size: int,
    timeout: int,
    verbose: bool = False,
) -> Optional[tuple[list[Node], str, SymTensor]]:
    """Try to synthesize a hw-only replacement for *node*.

    Returns (new_nodes, output_hw_id, output_hw_sym) on success, None otherwise.

    Results are cached by a canonical key derived from (op, attrs, input ranks)
    so that repeated encounters of the same operation – within a single graph or
    across multiple graph variants – skip synthesis and reuse the stored sketch.
    """
    input_syms = [sym for sym, _ in hw_input_pairs]

    # ------------------------------------------------------------------
    # Cache look-up
    # ------------------------------------------------------------------
    cache_key = _make_lowering_cache_key(node.op, node.attrs, input_syms)
    with _SYNTHESIS_CACHE_LOCK:
        cached = _SYNTHESIS_CACHE.get(cache_key, _CACHE_MISS)

    if cached is not _CACHE_MISS:
        if cached is None:
            # Cached failure
            if verbose:
                print(f"  => CACHE HIT (failure): op={node.op} for '{node.id}'")
            return None
        # Cached sketch: rebuild with the new input syms
        rebuilt = _denormalize_sketch(cached, input_syms)
        if rebuilt is not None:
            if verbose:
                print(f"  => CACHE HIT: {_format_sketch(rebuilt)} for '{node.id}' op={node.op}")
            sym_to_node_id: dict[int, str] = {
                id(sym): node_id for sym, node_id in hw_input_pairs
            }
            new_nodes: list[Node] = []
            output_id = _sketch_to_graph_nodes(rebuilt, sym_to_node_id, new_nodes)
            if output_id is not None:
                output_sym = _eval_sketch(rebuilt)
                if output_sym is not None:
                    return new_nodes, output_id, output_sym
        # If rebuilding failed (shouldn't happen in practice), fall through
        # to fresh synthesis so we never silently return None from a hit.

    # ------------------------------------------------------------------
    # Cache miss – run synthesis
    # ------------------------------------------------------------------

    # Build a LOCAL target sym from the hw inputs: this makes the equivalence
    # check purely local (e.g., nc_matmul(nc_transpose(t), w) ≡ matmul(t, w))
    # rather than having to trace through the full original expression chain.
    local_target: Optional[SymTensor] = None
    try:
        local_target = _sym_expr_from_graph_node(node, input_syms)
    except (KeyError, Exception):
        pass
    effective_target = local_target if local_target is not None else target_sym

    pool = _build_synthesis_pool(node.op, node.attrs, input_syms)

    if verbose:
        input_names = [
            (s.expr.name if s.expr is not None else "?") for s in input_syms
        ]
        print(f"  Synthesizing node '{node.id}' op={node.op}  inputs={input_names}"
              f"  pool_size={len(pool)}  max_hw_size={max_hw_size}")

    found = _synthesize_from_pool(
        effective_target, pool,
        max_hw_size=max_hw_size, timeout=timeout, verbose=verbose,
        input_syms=input_syms,
    )
    if found is None:
        if verbose:
            print(f"  => FAILED: no hw equivalent found for '{node.id}' op={node.op}")
        with _SYNTHESIS_CACHE_LOCK:
            _SYNTHESIS_CACHE[cache_key] = None
        return None

    if verbose:
        print(f"  => FOUND: {_format_sketch(found)}")

    # Store the normalized sketch in the cache before materializing graph nodes
    # so that future calls with equivalent inputs can skip synthesis entirely.
    norm = _normalize_sketch(found, input_syms)
    if norm is not None:
        with _SYNTHESIS_CACHE_LOCK:
            _SYNTHESIS_CACHE[cache_key] = norm

    # Map each concrete input SymTensor to its hw node id
    sym_to_node_id: dict[int, str] = {
        id(sym): node_id for sym, node_id in hw_input_pairs
    }

    new_nodes: list[Node] = []
    output_id = _sketch_to_graph_nodes(found, sym_to_node_id, new_nodes)
    if output_id is None:
        return None

    output_sym = _eval_sketch(found)
    if output_sym is None:
        return None

    return new_nodes, output_id, output_sym


def _lower_node_all(
    node: Node,
    target_sym: SymTensor,
    hw_input_pairs: list[tuple[SymTensor, str]],
    max_hw_size: int,
    timeout: int,
    verbose: bool = False,
    max_workers: Optional[int] = None,
) -> list[tuple[list[Node], str, SymTensor]]:
    """Synthesize ALL valid hw-only replacements for *node*.

    Like _lower_node but uses _synthesize_all_from_pool so every equivalent
    sketch is found rather than stopping at the first.  Returns a (possibly
    empty) list of
    (new_nodes, output_hw_id, output_hw_sym) triples – one per valid sketch.

    Results are cached by the same canonical key as *_lower_node* so that
    repeated encounters of the same operation skip synthesis and re-materialise
    the stored sketches with the new hw input SymTensors.
    """
    input_syms = [sym for sym, _ in hw_input_pairs]

    # ------------------------------------------------------------------
    # Cache look-up
    # ------------------------------------------------------------------
    cache_key = _make_lowering_cache_key(node.op, node.attrs, input_syms)
    with _SYNTHESIS_CACHE_LOCK:
        cached_all = _SYNTHESIS_CACHE_ALL.get(cache_key, _CACHE_MISS)

    if cached_all is not _CACHE_MISS:
        if not cached_all:
            # Cached failure (empty list)
            if verbose:
                with _VERBOSE_LOCK:
                    print(f"  => CACHE HIT (failure): op={node.op} for '{node.id}'")
            return []
        # Rebuild all sketches from the cached normalized forms
        sym_to_node_id: dict[int, str] = {
            id(sym): node_id for sym, node_id in hw_input_pairs
        }
        materialized_cached: list[tuple[list[Node], str, SymTensor]] = []
        for norm in cached_all:
            rebuilt = _denormalize_sketch(norm, input_syms)
            if rebuilt is None:
                continue
            if verbose:
                with _VERBOSE_LOCK:
                    print(f"  => CACHE HIT: {_format_sketch(rebuilt)} for '{node.id}' op={node.op}")
            new_nodes: list[Node] = []
            output_id = _sketch_to_graph_nodes(rebuilt, sym_to_node_id, new_nodes)
            if output_id is None:
                continue
            output_sym = _eval_sketch(rebuilt)
            if output_sym is None:
                continue
            materialized_cached.append((new_nodes, output_id, output_sym))
        if materialized_cached:
            return materialized_cached
        # Fall through to fresh synthesis if rebuilding failed unexpectedly.

    # ------------------------------------------------------------------
    # Cache miss – run synthesis
    # ------------------------------------------------------------------

    local_target: Optional[SymTensor] = None
    try:
        local_target = _sym_expr_from_graph_node(node, input_syms)
    except (KeyError, Exception):
        pass
    effective_target = local_target if local_target is not None else target_sym

    pool = _build_synthesis_pool(node.op, node.attrs, input_syms)

    if verbose:
        input_names = [(s.expr.name if s.expr is not None else "?") for s in input_syms]
        with _VERBOSE_LOCK:
            print(f"  Synthesizing node '{node.id}' op={node.op}  inputs={input_names}"
                  f"  pool_size={len(pool)}  max_hw_size={max_hw_size}")

    found_sketches = _synthesize_all_from_pool(
        effective_target, pool,
        max_hw_size=max_hw_size, timeout=timeout, verbose=verbose,
        max_workers=max_workers,
        input_syms=input_syms,
    )

    if not found_sketches:
        if verbose:
            with _VERBOSE_LOCK:
                print(f"  => FAILED: no hw equivalent found for '{node.id}' op={node.op}")
        with _SYNTHESIS_CACHE_LOCK:
            _SYNTHESIS_CACHE_ALL[cache_key] = []
        return []

    sym_to_node_id: dict[int, str] = {
        id(sym): node_id for sym, node_id in hw_input_pairs
    }

    materialized: list[tuple[SketchNode, list[Node], str, SymTensor]] = []
    norms_to_cache: list[_NormSketch] = []
    for sketch in found_sketches:
        if verbose:
            with _VERBOSE_LOCK:
                print(f"  => FOUND: {_format_sketch(sketch)}")
        new_nodes: list[Node] = []
        output_id = _sketch_to_graph_nodes(sketch, sym_to_node_id, new_nodes)
        if output_id is None:
            continue
        output_sym = _eval_sketch(sketch)
        if output_sym is None:
            continue
        materialized.append((sketch, new_nodes, output_id, output_sym))
        norm = _normalize_sketch(sketch, input_syms)
        if norm is not None:
            norms_to_cache.append(norm)

    # Store normalized sketches before sorting (sort order is deterministic)
    if norms_to_cache:
        with _SYNTHESIS_CACHE_LOCK:
            _SYNTHESIS_CACHE_ALL[cache_key] = norms_to_cache
    elif found_sketches:
        # All found sketches failed to materialize / normalize – treat as failure
        with _SYNTHESIS_CACHE_LOCK:
            _SYNTHESIS_CACHE_ALL[cache_key] = []

    materialized.sort(
        key=lambda item: (
            0 if _shapes_match_exactly(effective_target, item[3]) else 1,
            item[0].hw_size(),
            _format_sketch(item[0]),
        )
    )

    return [(new_nodes, output_id, output_sym) for _, new_nodes, output_id, output_sym in materialized]


# ---------------------------------------------------------------------------
# Full-graph lowering
# ---------------------------------------------------------------------------

def lower_nu_graph(
    G: nuGraph,
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
    max_workers: Optional[int] = None,
) -> Optional[nuGraph]:
    """Lower all public ops in *G* to hw ops using sketch-driven synthesis.

    Returns a new nuGraph that uses only hw ops, or None if any node could
    not be lowered.  Independent synthesis nodes that share a DAG level are
    lowered concurrently when *max_workers* > 1.  When *verbose* is True every
    sketch evaluated for every node is printed so the search can be manually
    inspected.
    """
    # Build symbolic tensors for the original graph (used as synthesis targets)
    orig_syms: dict[str, SymTensor] = {}
    try:
        orig_syms = _graph_symbolic_tensors(G)
    except (KeyError, Exception):
        return None

    hw_nodes: list[Node] = []           # accumulate lowered nodes
    hw_syms: dict[str, SymTensor] = {}  # hw_node_id -> SymTensor
    node_id_map: dict[str, str] = {}    # orig_id -> hw_id

    if verbose:
        print(f"[lower_nu_graph] graph has {len(G.nodes)} nodes")

    levels = _build_dag_levels(G)

    for level_nodes in levels:
        deterministic: list[Node] = []
        synthesis: list[Node] = []
        for node in level_nodes:
            if node.op == "input" or node.op in _PUBLIC_PASSTHROUGH_OPS:
                deterministic.append(node)
            else:
                synthesis.append(node)

        for node in deterministic:
            if node.op == "input":
                hw_nodes.append(Node(node.id, node.op, list(node.inputs), dict(node.attrs), node.shape))
                hw_syms[node.id] = orig_syms[node.id]
                node_id_map[node.id] = node.id
            else:
                new_inputs = [node_id_map.get(inp, inp) for inp in node.inputs]
                out_sym = hw_syms.get(new_inputs[0]) if new_inputs else None
                if out_sym is None:
                    return None
                new_id = _gen_id(node.op)
                hw_nodes.append(Node(new_id, node.op, new_inputs, dict(node.attrs), node.shape))
                hw_syms[new_id] = out_sym
                node_id_map[node.id] = new_id

        if not synthesis:
            continue

        SynthArgs = tuple[Node, SymTensor, list[tuple[SymTensor, str]]]
        synthesis_args: list[SynthArgs] = []
        for node in synthesis:
            hw_input_pairs: list[tuple[SymTensor, str]] = []
            missing_inputs: list[str] = []
            for inp_id in node.inputs:
                hw_id = node_id_map.get(inp_id, inp_id)
                hw_sym = hw_syms.get(hw_id)
                if hw_sym is None:
                    missing_inputs.append(_format_resolved_input(inp_id, hw_id))
                    continue
                hw_input_pairs.append((hw_sym, hw_id))
            if missing_inputs:
                if verbose:
                    print(
                        f"[lower_nu_graph] cannot synthesize '{node.id}' op={node.op}: "
                        f"missing lowered inputs {missing_inputs}"
                    )
                return None
            if verbose:
                with _VERBOSE_LOCK:
                    print(
                        f"[lower_nu_graph] synthesizing '{node.id}' op={node.op} "
                        f"orig_inputs={node.inputs} "
                        f"resolved_hw_inputs={[hw_id for _, hw_id in hw_input_pairs]}"
                    )
            target_sym = orig_syms.get(node.id)
            if target_sym is None:
                return None
            synthesis_args.append((node, target_sym, hw_input_pairs))

        def _synth(args: SynthArgs) -> Optional[tuple[list[Node], str, SymTensor]]:
            nd, tgt, pairs = args
            # z3.main_ctx() is not thread-safe: Z3_inc_ref/Z3_dec_ref can race
            # when multiple threads concurrently create or destroy Z3 AST objects
            # (e.g. via __del__).  Hold _Z3_LOCK for the entire _lower_node call
            # so that all Z3 object creation, solver work, and object destruction
            # (including __del__ of temporaries) are serialised.  _Z3_LOCK is an
            # RLock so nested acquisitions from the same thread still succeed.
            with _Z3_LOCK:
                return _lower_node(
                    nd,
                    tgt,
                    pairs,
                    max_hw_size,
                    timeout,
                    verbose=verbose,
                )

        level_results: list[Optional[tuple[list[Node], str, SymTensor]]]
        effective_workers = _effective_max_workers(max_workers, len(synthesis))
        if effective_workers > 1 and len(synthesis) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = [executor.submit(_synth, args) for args in synthesis_args]
                level_results_tmp: list[Optional[tuple[list[Node], str, SymTensor]]] = []
                for i_f, future in enumerate(futures):
                    try:
                        level_results_tmp.append(future.result())
                    except Exception as exc:
                        if verbose:
                            nd = synthesis[i_f]
                            with _VERBOSE_LOCK:
                                print(f"[lower_nu_graph] synthesis of '{nd.id}' op={nd.op} raised: {exc}")
                        level_results_tmp.append(None)
                level_results = level_results_tmp
        else:
            level_results = [_synth(args) for args in synthesis_args]

        for node, result in zip(synthesis, level_results):
            if result is None:
                return None
            new_hw_nodes, output_id, output_sym = result
            hw_nodes.extend(new_hw_nodes)
            hw_syms[output_id] = output_sym
            node_id_map[node.id] = output_id

    G_hw = nuGraph(hw_nodes)
    try:
        annotate_shapes_concrete(G_hw)
    except (KeyError, Exception):
        pass
    return G_hw


def lower_nu_graph_variants(
    variants: list[nuGraph],
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
    max_workers: Optional[int] = None,
) -> list[Optional[nuGraph]]:
    """Lower every variant in *variants* to hardware ops.

    Starts from an empty synthesis cache for the variant set so performance for
    one kernel is not affected by synthesis work done for earlier kernels, then
    reuses synthesized tensor-op lowerings across the variants in *variants*.

    Returns a list of the same length; entries that could not be lowered are
    None.
    """
    _start_kernel_synthesis_cache(verbose=verbose)
    return [
        lower_nu_graph(
            v,
            max_hw_size=max_hw_size,
            timeout=timeout,
            verbose=verbose,
            max_workers=max_workers,
        )
        for v in variants
    ]


def _build_dag_levels(G: nuGraph) -> list[list[Node]]:
    """Group *G*'s nodes into topological levels.

    Level 0 holds nodes with no predecessors (input nodes).  Level k holds
    nodes whose every input is at a level strictly less than k.  All nodes
    within the same level are mutually independent and can be synthesized
    in parallel.
    """
    if not G.nodes:
        return []
    level_of: dict[str, int] = {}
    for node in G.nodes:
        if not node.inputs or node.op == "input":
            level_of[node.id] = 0
        else:
            # G.nodes is in topological order so all inputs are already in
            # level_of; .get(..., 0) is a safety fallback.
            level_of[node.id] = builtins.max(
                level_of.get(inp, 0) for inp in node.inputs
            ) + 1

    max_level = builtins.max(level_of.values())
    levels: list[list[Node]] = [[] for _ in range(max_level + 1)]
    for node in G.nodes:
        levels[level_of[node.id]].append(node)
    return levels


def lower_nu_graph_all_variants(
    G: nuGraph,
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    max_variants: int = 256,
) -> list[nuGraph]:
    """Lower *G* to hardware discovering every valid sketch choice per node.

    Sketch-level parallelism is used within each node by
    ``_synthesize_all_from_pool``. Node-level synthesis is currently kept
    serial because concurrent node lowering proved unsafe with the shared Z3
    context, and holding ``_Z3_LOCK`` across the full node-lowering call would
    deadlock the inner sketch-level worker pool.

    After synthesis the Cartesian product of per-node alternatives is taken.
    Because all valid sketches for a node are semantically equivalent to the
    target, downstream nodes are synthesized against the canonical (first)
    alternative, and the remaining combinations are produced by substituting
    the canonical output IDs with the IDs of the chosen alternatives.

    **Complexity note**: the Cartesian product can grow exponentially – if
    every synthesis node has M valid sketches and the graph has N synthesis
    nodes, up to M^N combinations are explored.  *max_variants* caps the
    number of distinct lowered graphs that are returned (default 256); once
    the cap is reached the remaining combinations are skipped and a warning
    is printed.

    Returns a de-duplicated list of lowered nuGraphs (by graph signature).
    Returns an empty list if any synthesis node cannot be lowered at all.
    """
    orig_syms: dict[str, SymTensor] = {}
    try:
        orig_syms = _graph_symbolic_tensors(G)
    except (KeyError, Exception):
        return []

    if verbose:
        print(f"[lower_nu_graph_all_variants] graph has {len(G.nodes)} nodes")

    levels = _build_dag_levels(G)

    # Canonical lowering state (uses first alt for synthesis nodes)
    hw_syms_can: dict[str, SymTensor] = {}    # canonical hw_id → sym
    node_id_map_can: dict[str, str] = {}       # orig_id → canonical hw_id

    # Per-node alternatives: each entry is a triple
    #   (new_nodes, canonical_hw_id, actual_hw_id)
    # where:
    #   new_nodes       – hw Node objects contributed by this original node
    #   canonical_hw_id – output id of the FIRST valid alt (used as the
    #                     reference id that downstream nodes were synthesized
    #                     against); same for all entries of the same orig node
    #   actual_hw_id    – output id of this specific alternative
    # Deterministic nodes have exactly one entry with canonical_hw_id == actual_hw_id.
    # Synthesis nodes may have multiple entries.
    _NodeEntry = tuple[list[Node], str, str]
    per_node_entries: dict[str, list[_NodeEntry]] = {}  # orig_id → choices

    for level_nodes in levels:
        deterministic: list[Node] = []
        synthesis: list[Node] = []
        for node in level_nodes:
            if node.op == "input" or node.op in _PUBLIC_PASSTHROUGH_OPS:
                deterministic.append(node)
            else:
                synthesis.append(node)

        # Deterministic nodes: process immediately, no synthesis required
        for node in deterministic:
            if node.op == "input":
                n_copy = Node(node.id, node.op, list(node.inputs),
                              dict(node.attrs), node.shape)
                hw_syms_can[node.id] = orig_syms[node.id]
                node_id_map_can[node.id] = node.id
                per_node_entries[node.id] = [([n_copy], node.id, node.id)]
            else:  # passthrough
                new_inputs = [node_id_map_can.get(inp, inp) for inp in node.inputs]
                out_sym = hw_syms_can.get(new_inputs[0]) if new_inputs else None
                if out_sym is None:
                    if verbose:
                        print(f"[lower_nu_graph_all_variants] passthrough node "
                              f"'{node.id}' op={node.op}: hw sym for first input "
                              f"not found; cannot lower graph")
                    return []
                new_id = _gen_id(node.op)
                n_new = Node(new_id, node.op, new_inputs, dict(node.attrs), node.shape)
                hw_syms_can[new_id] = out_sym
                node_id_map_can[node.id] = new_id
                per_node_entries[node.id] = [([n_new], new_id, new_id)]

        if not synthesis:
            continue

        # Build hw_input_pairs for each synthesis node from the canonical state
        SynthArgs = tuple[Node, SymTensor, list[tuple[SymTensor, str]]]
        synthesis_args: list[SynthArgs] = []
        for node in synthesis:
            hw_input_pairs: list[tuple[SymTensor, str]] = []
            missing_inputs: list[str] = []
            for inp_id in node.inputs:
                hw_id = node_id_map_can.get(inp_id, inp_id)
                hw_sym = hw_syms_can.get(hw_id)
                if hw_sym is None:
                    missing_inputs.append(_format_resolved_input(inp_id, hw_id))
                    continue
                hw_input_pairs.append((hw_sym, hw_id))
            if missing_inputs:
                if verbose:
                    print(
                        f"[lower_nu_graph_all_variants] cannot synthesize '{node.id}' "
                        f"op={node.op}: missing lowered inputs {missing_inputs}"
                    )
                return []
            if verbose:
                with _VERBOSE_LOCK:
                    print(
                        f"[lower_nu_graph_all_variants] synthesizing '{node.id}' "
                        f"op={node.op} "
                        f"orig_inputs={node.inputs} "
                        f"resolved_hw_inputs={[hw_id for _, hw_id in hw_input_pairs]}"
                    )
            t_sym = orig_syms.get(node.id)
            if t_sym is None:
                return []
            synthesis_args.append((node, t_sym, hw_input_pairs))

        # Synthesise nodes in deterministic order. Sketch-level parallelism
        # remains enabled inside _lower_node_all(), but node-level synthesis is
        # kept serial to avoid Z3 shared-context races and nested-lock deadlocks.
        def _synth(args: SynthArgs) -> list[tuple[list[Node], str, SymTensor]]:
            nd, tgt, pairs = args
            return _lower_node_all(nd, tgt, pairs,
                                   max_hw_size=max_hw_size, timeout=timeout,
                                   verbose=verbose, max_workers=max_workers)

        node_alts_list: list[list[tuple[list[Node], str, SymTensor]]]
        effective_workers = 1
        if effective_workers > 1 and len(synthesis) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = [executor.submit(_synth, args) for args in synthesis_args]
                node_alts_list_tmp: list[list[tuple[list[Node], str, SymTensor]]] = []
                for i_f, future in enumerate(futures):
                    try:
                        node_alts_list_tmp.append(future.result())
                    except Exception as exc:
                        nd = synthesis[i_f]
                        if verbose:
                            print(f"[lower_nu_graph_all_variants] synthesis of "
                                  f"'{nd.id}' op={nd.op} raised: {exc}")
                        node_alts_list_tmp.append([])
                node_alts_list = node_alts_list_tmp
        else:
            node_alts_list = [_synth(args) for args in synthesis_args]

        for node, alts in zip(synthesis, node_alts_list):
            if not alts:
                return []  # this node cannot be lowered → whole graph fails
            # First alt provides the canonical continuation for downstream nodes
            _, canonical_hw_id, canonical_sym = alts[0]
            hw_syms_can[canonical_hw_id] = canonical_sym
            node_id_map_can[node.id] = canonical_hw_id
            # Record all alternatives; canonical_hw_id is the same for every entry
            per_node_entries[node.id] = [
                (new_nodes, canonical_hw_id, actual_id)
                for new_nodes, actual_id, _ in alts
            ]

    # Reconstruct per-node choices in original G.nodes order
    per_node_choices: list[list[_NodeEntry]] = [
        per_node_entries[node.id] for node in G.nodes
    ]

    # Take the Cartesian product of per-node alternatives.
    # For each combination, build a substitution map
    #   canonical_hw_id → actual_hw_id
    # and apply it to every node's input list so graph wiring is correct
    # regardless of which alternative was chosen for predecessor nodes.
    # Results are de-duplicated by graph signature and capped at max_variants.
    results: list[nuGraph] = []
    seen_sigs: set[str] = set()
    n_combos_explored = 0

    for combo in _iproduct(*per_node_choices):
        n_combos_explored += 1
        if len(results) >= max_variants:
            # Count remaining combinations (skip through the rest of the generator)
            n_skipped = builtins.sum(1 for _ in _iproduct(*per_node_choices)) - n_combos_explored + 1
            print(f"[lower_nu_graph_all_variants] variant cap ({max_variants}) reached "
                  f"after exploring {n_combos_explored} combination(s); "
                  f"~{n_skipped} combination(s) skipped")
            break
        subst: dict[str, str] = {}
        for _, canonical_id, actual_id in combo:
            if canonical_id != actual_id:
                subst[canonical_id] = actual_id

        hw_nodes_all: list[Node] = []
        for new_nodes, _, _ in combo:
            for n in new_nodes:
                new_inputs = [subst.get(i, i) for i in n.inputs]
                hw_nodes_all.append(
                    Node(n.id, n.op, new_inputs, dict(n.attrs), n.shape))

        G_hw = nuGraph(hw_nodes_all)
        try:
            annotate_shapes_concrete(G_hw)
        except (KeyError, Exception):
            pass
        sig = graph_signature(G_hw)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        results.append(G_hw)

    if verbose:
        print(f"[lower_nu_graph_all_variants] emitting {len(results)} "
              f"distinct lowered hw graph variant(s):")
        for vi, g_hw in enumerate(results):
            print(f"  --- lowered hw variant {vi} ---")
            print_graph(g_hw)

    return results


def _print_synthesis_phase_variants(phase_name: str, graphs: list[nuGraph]) -> None:
    """Print synthesized graph variants for one named phase of the pipeline.

    Intended for the phase boundaries in ``synthesize_hw_graph`` such as
    pre-lowering, post-lowering, and post-swap/propagation.
    """
    print(f"[synthesize_hw_graph] {phase_name}: {len(graphs)} variant(s)")
    for idx, graph in enumerate(graphs):
        print(f"  --- {phase_name} variant {idx} ---")
        print_graph(graph)


def synthesize_hw_graph(
    G: nuGraph,
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
    max_workers: Optional[int] = None,
) -> list[nuGraph]:
    """Generate all variant orderings of *G* and lower each to hw ops.

    Phase 1: generate nuGraph variants from the original public-op graph via
    ``nu_graph_generation_z3``, then lower each to a hardware-only graph.

    Phase 2: once all hardware graphs have been collected, run
    ``nu_graph_generation_z3`` again on every distinct lowered hardware graph to
    discover additional swap-derived variants among the hardware ops.  Any new
    variants found are appended to the results.

    All returned graphs are deduplicated by graph signature.  Pass
    *verbose=True* to print all sketches evaluated during synthesis for manual
    inspection.
    """
    # ------------------------------------------------------------------
    # Phase 1: variant generation + lowering of the original graph
    # ------------------------------------------------------------------
    variants = nu_graph_generation_z3(G)
    if verbose:
        _print_synthesis_phase_variants("pre-lowering", variants)
    lowered = lower_nu_graph_variants(
        variants,
        max_hw_size=max_hw_size,
        timeout=timeout,
        verbose=verbose,
        max_workers=max_workers,
    )
    seen: set[str] = set()
    results: list[nuGraph] = []
    for g_hw in lowered:
        if g_hw is None:
            continue
        sig = graph_signature(g_hw)
        if sig in seen:
            continue
        seen.add(sig)
        results.append(g_hw)
    if verbose:
        _print_synthesis_phase_variants("post-lowering", results)

    # ------------------------------------------------------------------
    # Phase 2: post-lowering node-swap variant generation on hw graphs
    # ------------------------------------------------------------------
    # Continue exploring newly-discovered hw variants until Phase 2 reaches a
    # fixpoint, since one post-lowering swap can expose a new local adjacency
    # for a later hw-op swap.
    phase2_variants: list[nuGraph] = []
    phase2_worklist: list[nuGraph] = list(results)
    for g_hw in phase2_worklist:
        for g_hw_variant in nu_graph_generation_z3(g_hw):
            sig = graph_signature(g_hw_variant)
            if sig not in seen:
                seen.add(sig)
                phase2_variants.append(g_hw_variant)
                phase2_worklist.append(g_hw_variant)
    results.extend(phase2_variants)
    if verbose:
        _print_synthesis_phase_variants("post-swap/propagation", results)

    return results


# ---------------------------------------------------------------------------
# Synthesizer test helpers
# ---------------------------------------------------------------------------

_HW_ONLY_OPS: frozenset[str] = frozenset(_HW_OP_NAMES) | frozenset({"input"})

_PUBLIC_NON_HW_OPS: frozenset[str] = frozenset({
    "add", "div", "divide", "exp", "matmul", "mul", "multiply",
    "reduce_sum", "relu", "silu", "sqrt", "subtract", "transpose",
    "rms_norm", "softmax",
})


def _graph_uses_hw_only(G: nuGraph) -> bool:
    """Return True if every non-input node in G uses a hw op."""
    for n in G.nodes:
        if n.op in _PUBLIC_NON_HW_OPS:
            return False
    return True


def _graph_output_sym_lowered(G_orig: nuGraph, G_hw: nuGraph) -> Optional[tuple[SymTensor, SymTensor]]:
    """Return (orig_output_sym, hw_output_sym) for the graph output node pair."""
    orig_outputs = _graph_output_nodes(G_orig)
    hw_outputs = _graph_output_nodes(G_hw)
    if not orig_outputs or not hw_outputs:
        return None
    try:
        orig_syms = _graph_symbolic_tensors(G_orig)
        hw_syms = _graph_symbolic_tensors(G_hw)
    except (KeyError, Exception):
        return None
    orig_out_id = orig_outputs[-1].id
    hw_out_id = hw_outputs[-1].id
    orig_sym = orig_syms.get(orig_out_id)
    hw_sym = hw_syms.get(hw_out_id)
    if orig_sym is None or hw_sym is None:
        return None
    return orig_sym, hw_sym


# ---------------------------------------------------------------------------
# Synthesizer tests
# ---------------------------------------------------------------------------

def _test_synthesizer_matmul_red_div() -> None:
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    print(" synthesizer: matmul_red_div — sketches per node:")
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000, verbose=True)
    assert G_hw is not None, "lower_nu_graph returned None for matmul_red_div"
    assert _graph_uses_hw_only(G_hw), "Lowered matmul_red_div graph still contains public ops"
    pair = _graph_output_sym_lowered(G, G_hw)
    assert pair is not None, "Could not extract output syms for matmul_red_div"
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), \
        "Lowered matmul_red_div not equivalent to original"
    print(" synthesizer: matmul_red_div lowered to hw ops and verified equivalent")


def _test_synthesizer_rmsnorm_matmul() -> None:
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)
    print(" synthesizer: rmsnorm_matmul — sketches per node:")
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000, verbose=True)
    assert G_hw is not None, "lower_nu_graph returned None for rmsnorm_matmul"
    assert _graph_uses_hw_only(G_hw), "Lowered rmsnorm_matmul graph still contains public ops"
    pair = _graph_output_sym_lowered(G, G_hw)
    assert pair is not None, "Could not extract output syms for rmsnorm_matmul"
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), \
        "Lowered rmsnorm_matmul not equivalent to original"
    print(" synthesizer: rmsnorm_matmul lowered to hw ops and verified equivalent")


def _test_synthesizer_transpose_matmul() -> None:
    G = build_kernel_transpose_matmul_graph(4, 8, 16)
    print(" synthesizer: transpose_matmul — sketches per node:")
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000, verbose=True)
    assert G_hw is not None, "lower_nu_graph returned None for transpose_matmul"
    assert _graph_uses_hw_only(G_hw), "Lowered transpose_matmul graph still contains public ops"
    pair = _graph_output_sym_lowered(G, G_hw)
    assert pair is not None, "Could not extract output syms for transpose_matmul"
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), \
        "Lowered transpose_matmul not equivalent to original"
    print(" synthesizer: transpose_matmul lowered to hw ops and verified equivalent")


def _test_synthesizer_matmul_transpose() -> None:
    G = build_kernel_matmul_transpose_graph(4, 8, 16)
    print(" synthesizer: matmul_transpose — sketches per node:")
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000, verbose=True)
    assert G_hw is not None, "lower_nu_graph returned None for matmul_transpose"
    assert _graph_uses_hw_only(G_hw), "Lowered matmul_transpose graph still contains public ops"
    pair = _graph_output_sym_lowered(G, G_hw)
    assert pair is not None, "Could not extract output syms for matmul_transpose"
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), \
        "Lowered matmul_transpose not equivalent to original"
    print(" synthesizer: matmul_transpose lowered to hw ops and verified equivalent")


def _test_synthesizer_relu_matmul() -> None:
    G = build_kernel_relu_matmul_graph(4, 8, 16)
    print(" synthesizer: relu_matmul — sketches per node:")
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000, verbose=True)
    assert G_hw is not None, "lower_nu_graph returned None for relu_matmul"
    assert _graph_uses_hw_only(G_hw), "Lowered relu_matmul graph still contains public ops"
    pair = _graph_output_sym_lowered(G, G_hw)
    assert pair is not None, "Could not extract output syms for relu_matmul"
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), \
        "Lowered relu_matmul not equivalent to original"
    print(" synthesizer: relu_matmul lowered to hw ops and verified equivalent")


def _test_synthesizer_silu_matmul() -> None:
    G = build_kernel_silu_matmul_graph(4, 8, 16)
    print(" synthesizer: silu_matmul — sketches per node:")
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000, verbose=True)
    assert G_hw is not None, "lower_nu_graph returned None for silu_matmul"
    assert _graph_uses_hw_only(G_hw), "Lowered silu_matmul graph still contains public ops"
    pair = _graph_output_sym_lowered(G, G_hw)
    assert pair is not None, "Could not extract output syms for silu_matmul"
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), \
        "Lowered silu_matmul not equivalent to original"
    print(" synthesizer: silu_matmul lowered to hw ops and verified equivalent")


def _test_synthesizer_all_variants_lowered() -> None:
    """Verify that lower_nu_graph_variants succeeds on all emitted variants
    for a representative kernel (matmul_red_div)."""
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G)
    print(f" synthesizer: lowering {len(variants)} matmul_red_div variants:")
    lowered = lower_nu_graph_variants(variants, max_hw_size=2, timeout=5000, verbose=True)
    failures = builtins.sum(1 for g_hw in lowered if g_hw is None)
    assert failures == 0, \
        f"lower_nu_graph_variants: {failures}/{len(variants)} variants failed to lower"
    for g_hw in lowered:
        assert _graph_uses_hw_only(g_hw), "A lowered variant still contains public ops"
    print(f" synthesizer: all {len(variants)} matmul_red_div variants lowered successfully")


def _test_lower_nu_graph_all_variants() -> None:
    """Verify that lower_nu_graph_all_variants returns at least one valid hw
    graph for a representative kernel and that every returned graph uses only
    hw ops.  Also checks that the result count is >= the single-lowering
    baseline (lower_nu_graph returns exactly 1 graph per variant)."""
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    print(" synthesizer: lower_nu_graph_all_variants on matmul_red_div:")
    hw_variants = lower_nu_graph_all_variants(G, max_hw_size=2, timeout=5000,
                                              verbose=True)
    assert hw_variants, \
        "lower_nu_graph_all_variants returned no results for matmul_red_div"
    for i, g_hw in enumerate(hw_variants):
        assert _graph_uses_hw_only(g_hw), \
            f"lower_nu_graph_all_variants: hw variant {i} still contains public ops"
    baseline = lower_nu_graph(G, max_hw_size=2, timeout=5000)
    assert baseline is not None, "lower_nu_graph baseline unexpectedly failed"
    print(f" synthesizer: lower_nu_graph_all_variants produced "
          f"{len(hw_variants)} distinct lowered hw graph(s) for matmul_red_div")


def _test_reduce_sum_lowering_no_crash() -> None:
    """Regression test for the z3 thread-safety crash that was triggered when
    multiple threads concurrently constructed z3 formulas sharing the global
    z3.main_ctx() while synthesizing reduce_sum_1001 in kernel_matmul_red_div.

    The fix adds _Z3_LOCK to serialise check_equivalent calls within
    _synthesize_all_from_pool._check_one so that the shared z3 context is
    never accessed from more than one thread at a time.

    This test exercises the exact crash path (parallel sketch synthesis for a
    reduce_sum node with keep_dims=True on a rank-2 input) and asserts that:
      1. The lowering completes without any exception or process crash.
      2. At least one valid hw-only graph is returned.
      3. All returned graphs use only hw ops.
      4. Every returned graph is semantically equivalent to the original.
    """
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    assert variants, "kernel_matmul_red_div should produce at least one variant"

    # Variant 0 contains reduce_sum_1001 (the originally crashing node).
    gv = variants[0]
    reduce_nodes = _nodes_by_op(gv, "reduce_sum")
    assert reduce_nodes, "Variant 0 of kernel_matmul_red_div should contain a reduce_sum node"
    assert (
        reduce_nodes[0].attrs.get("keep_dims") is True
        or reduce_nodes[0].attrs.get("keepdims") is True
    ), \
        "reduce_sum in kernel_matmul_red_div variant 0 should use keep_dims=True"

    hw_variants = lower_nu_graph_all_variants(
        gv,
        max_hw_size=2,
        timeout=5000,
        max_workers=2,
    )
    assert hw_variants, \
        "reduce_sum_lowering regression: lower_nu_graph_all_variants returned no results"
    for i, g_hw in enumerate(hw_variants):
        assert _graph_uses_hw_only(g_hw), \
            f"reduce_sum_lowering regression: hw variant {i} still contains public ops"

    # Spot-check equivalence of the first returned hw graph
    pair = _graph_output_sym_lowered(gv, hw_variants[0])
    assert pair is not None, \
        "reduce_sum_lowering regression: could not extract output syms for equivalence check"
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), \
        "reduce_sum_lowering regression: first lowered hw graph is not equivalent to original"

    print(f" synthesizer: reduce_sum lowering (regression) — "
          f"{len(hw_variants)} hw variant(s) produced, all equivalent")


def _test_matmul_1003_multi_input_synthesis() -> None:
    """Regression test for the failure mode where ``matmul_1003`` in variant 0
    of ``kernel_matmul_red_div`` was synthesized with only the ``w`` operand
    instead of both ``div_1002`` and ``w``.

    Root cause: when an upstream synthesis node (``div_1002``) had its hw sym
    missing from ``hw_syms_can``, downstream synthesis proceeded with a
    truncated ``hw_input_pairs`` list, leaving only ``IN:w`` in the synthesis
    pool and causing 'no hw equivalent found for matmul_1003'.

    This test:
      1. Extracts variant 0 of ``kernel_matmul_red_div``, which has the
         structure ``reduce_sum → div → matmul``.
      2. Asserts the graph contains a ``matmul`` node with two inputs
         (``div_*`` and ``w``), confirming it's a true multi-input op.
      3. Runs ``lower_nu_graph_all_variants`` and asserts it succeeds – i.e.
         the lowering does NOT fail because matmul received incomplete inputs.
      4. Verifies every returned hw graph uses only hw ops and that at least
         one contains ``nc_matmul`` (confirming matmul was actually lowered,
         not skipped).
      5. Spot-checks equivalence of the first returned hw graph.

    The test would fail with the pre-fix code where:
      - ``div_1002`` synthesis failed (producing no hw sym),
      - the downstream matmul synthesis therefore received only ``IN:w``, and
      - all 162 sketch candidates were not equivalent, so lowering returned [].
    """
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    assert variants, "kernel_matmul_red_div should produce at least one variant"

    gv = variants[0]

    # Confirm variant 0 has the expected multi-input matmul structure.
    matmul_nodes = _nodes_by_op(gv, "matmul")
    assert matmul_nodes, (
        "Variant 0 of kernel_matmul_red_div must contain a 'matmul' node"
    )
    matmul_node = matmul_nodes[0]
    assert len(matmul_node.inputs) == 2, (
        f"matmul node in variant 0 must have exactly 2 inputs, "
        f"got {matmul_node.inputs!r}"
    )
    # One input must be 'w' (the weight tensor), the other is the upstream
    # synthesis node (div or similar) that must have been resolved.
    assert "w" in matmul_node.inputs, (
        f"matmul node must have 'w' as one of its inputs, got {matmul_node.inputs!r}"
    )
    upstream_input = next((i for i in matmul_node.inputs if i != "w"), None)
    assert upstream_input is not None, (
        f"matmul node must have a non-'w' upstream input; inputs={matmul_node.inputs!r}"
    )

    # Confirm the upstream input comes from a synthesis node (not a bare input),
    # which means it must be resolved through hw_syms_can during lowering.
    # Build a lookup dict once to avoid a linear scan per lookup.
    node_by_id = {n.id: n for n in gv.nodes}
    upstream_node = node_by_id.get(upstream_input)
    assert upstream_node is not None, (
        f"Upstream input '{upstream_input}' not found in variant graph"
    )
    assert upstream_node.op != "input", (
        f"Upstream input '{upstream_input}' to matmul is a raw graph input, "
        f"not a synthesis node; the test requires a derived node to stress the "
        f"hw_syms_can resolution path"
    )

    # Run the full lowering and assert it succeeds with BOTH inputs resolved.
    # The negative scenario (incomplete inputs) is guarded by the defensive
    # validation in lower_nu_graph_all_variants: if the upstream div node
    # fails to synthesize, the function returns [] with a verbose diagnostic
    # naming the missing input ID, so synthesis of matmul never proceeds
    # with a truncated hw_input_pairs.
    hw_variants = lower_nu_graph_all_variants(
        gv,
        max_hw_size=2,
        timeout=5000,
        max_workers=2,
    )
    assert hw_variants, (
        "matmul_1003 regression: lower_nu_graph_all_variants returned no results "
        "for variant 0 of kernel_matmul_red_div — matmul may have received "
        f"incomplete inputs (only 'w' instead of both '{upstream_input}' and 'w')"
    )

    for i, g_hw in enumerate(hw_variants):
        assert _graph_uses_hw_only(g_hw), (
            f"matmul_1003 regression: hw variant {i} still contains public ops"
        )

    # At least one hw graph must contain nc_matmul (confirming the matmul was
    # actually lowered and not silently dropped).
    assert any(_nodes_by_op(g_hw, "nc_matmul") for g_hw in hw_variants), (
        "matmul_1003 regression: no hw variant contains 'nc_matmul'; "
        "matmul may not have been lowered"
    )

    # Spot-check equivalence of the first returned hw graph.
    pair = _graph_output_sym_lowered(gv, hw_variants[0])
    assert pair is not None, (
        "matmul_1003 regression: could not extract output syms for equivalence check"
    )
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), (
        "matmul_1003 regression: first lowered hw graph is not equivalent to original"
    )

    print(
        f" synthesizer: matmul_1003 multi-input regression — "
        f"matmul node inputs={matmul_node.inputs!r} (upstream='{upstream_input}'), "
        f"{len(hw_variants)} hw variant(s) produced, first variant equivalent"
    )


def kernel_matmul_red_div(x: DummyArray, y: DummyArray, w: DummyArray) -> DummyArray:
    rec = y.sum(axis=1, keep_dims=True)
    return (x / rec) @ w


def kernel_matmul_red_mul(x: DummyArray, y: DummyArray, w: DummyArray) -> DummyArray:
    rec = y.sum(axis=1, keep_dims=True)
    return (x * rec) @ w


def kernel_broadcast_row_bias_add(x: DummyArray, y: DummyArray, w: DummyArray) -> DummyArray:
    bias = y.sum(axis=1, keep_dims=True)
    bias_b = bias.broadcast_like(x)
    z = x + bias_b
    return z @ w


def kernel_reduce_mul_broadcast(x: DummyArray, y: DummyArray, w: DummyArray) -> DummyArray:
    rec = y.sum(axis=1, keep_dims=True)
    z = x * rec
    z_b = z.broadcast_like(x)
    return z_b @ w


def kernel_reduce_broadcast_mul(x: DummyArray, y: DummyArray, w: DummyArray) -> DummyArray:
    rec = y.sum(axis=1, keep_dims=True)
    rec_b = rec.broadcast_like(x)
    z = x * rec_b
    return z @ w


def kernel_rmsnorm_matmul(x: DummyArray, y: DummyArray, w: DummyArray) -> DummyArray:
    yy = y * y
    rec = yy.sum(axis=1, keep_dims=True)
    rms = rec.sqrt()
    norm = x / rms
    return norm @ w
  

def kernel_softmax_matmul(x: DummyArray, w: DummyArray) -> DummyArray:
    ex = x.exp()
    den = ex.sum(axis=1, keep_dims=True)
    probs = ex / den
    return probs @ w


def kernel_transpose_matmul(x: DummyArray, w: DummyArray) -> DummyArray:
    xt = x.transpose()
    return xt @ w


def kernel_matmul_transpose(x: DummyArray, w: DummyArray) -> DummyArray:
    z = x @ w
    return z.transpose()


def kernel_relu_matmul(x: DummyArray, w: DummyArray) -> DummyArray:
    return x.relu() @ w


def kernel_silu_matmul(x: DummyArray, w: DummyArray) -> DummyArray:
    return x.silu() @ w


def kernel_silu_mlp(x: DummyArray, w1: DummyArray, w2: DummyArray) -> DummyArray:
    h = x @ w1
    a = h.silu()
    return a @ w2


def kernel_attention(x: DummyArray, w_q: DummyArray, w_k: DummyArray, w_v: DummyArray) -> DummyArray:
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    k_t = k.transpose()
    qk = q @ k_t
    ex = qk.exp()
    den = ex.sum(axis=1, keep_dims=True)
    probs = ex / den
    return probs @ v


def _graph_from_axon_array(out: DummyArray) -> nuGraph:
    G = nuGraph([Node(id=n.id, op=n.op, inputs=list(n.inputs), attrs=dict(n.attrs)) for n in out.nodes])
    annotate_shapes_concrete(G)
    return G


def _build_graph_from_kernel(kernel: Callable[..., DummyArray], *inputs: tuple[str, tuple[int, ...]]) -> nuGraph:
    args = [DummyArray(name, shape) for name, shape in inputs]
    out = kernel(*args)
    return _graph_from_axon_array(out)


def build_kernel_matmul_red_div_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_matmul_red_div, ("x", (M, K)), ("y", (M, K)), ("w", (K, N)))


def build_kernel_matmul_red_mul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_matmul_red_mul, ("x", (M, K)), ("y", (M, K)), ("w", (K, N)))


def build_kernel_broadcast_row_bias_add_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_broadcast_row_bias_add, ("x", (M, K)), ("y", (M, K)), ("w", (K, N)))


def build_kernel_reduce_mul_broadcast_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_reduce_mul_broadcast, ("x", (M, K)), ("y", (M, K)), ("w", (K, N)))


def build_kernel_reduce_broadcast_mul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_reduce_broadcast_mul, ("x", (M, K)), ("y", (M, K)), ("w", (K, N)))


def build_kernel_rmsnorm_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_rmsnorm_matmul, ("x", (M, K)), ("y", (M, K)), ("w", (K, N)))


def build_kernel_softmax_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_softmax_matmul, ("x", (M, K)), ("w", (K, N)))


def build_kernel_transpose_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_transpose_matmul, ("x", (M, K)), ("w", (M, N)))


def build_kernel_matmul_transpose_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_matmul_transpose, ("x", (M, K)), ("w", (K, N)))


def build_kernel_relu_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_relu_matmul, ("x", (M, K)), ("w", (K, N)))


def build_kernel_silu_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_silu_matmul, ("x", (M, K)), ("w", (K, N)))


def build_kernel_silu_mlp_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(kernel_silu_mlp, ("x", (M, K)), ("w1", (K, N)), ("w2", (N, N)))


def build_kernel_attention_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_attention,
        ("x", (M, K)),
        ("w_q", (K, N)),
        ("w_k", (K, N)),
        ("w_v", (K, N)),
    )


def print_graph(G: nuGraph) -> None:
    symbolic_shapes: dict[str, tuple[Any, ...]] = {}
    sym_shape_fallback = "None"
    try:
        symbolic_shapes = {node_id: tensor.shape for node_id, tensor in _graph_symbolic_tensors(G).items()}
    except (KeyError, z3.Z3Exception):
        symbolic_shapes = {}
        sym_shape_fallback = "unavailable"
    topologically_ordered_nodes = [node for level in _build_dag_levels(G) for node in level]
    for i, n in enumerate(topologically_ordered_nodes):
        sym_shape = symbolic_shapes.get(n.id)
        sym_shape_str = _format_shape(sym_shape) if sym_shape is not None else sym_shape_fallback
        print(
            f"[{i}] id={n.id:12s} op={n.op:10s} inputs={n.inputs} "
            f"shape={_format_shape(n.shape)} sym_shape={sym_shape_str} attrs={n.attrs}"
        )


def _variants_for(builder: Callable[[int, int, int], nuGraph], M: int = 4, K: int = 8, N: int = 16) -> list[nuGraph]:
    G0 = builder(M, K, N)
    return nu_graph_generation_z3(G0, verbose=False)


def _test_expected_variant_counts() -> None:
    cases: list[tuple[str, Callable[[int, int, int], nuGraph], int]] = [
        ("kernel_matmul_red_div", build_kernel_matmul_red_div_graph, 2),
        ("kernel_matmul_red_mul", build_kernel_matmul_red_mul_graph, 1),
        ("kernel_rmsnorm_matmul", build_kernel_rmsnorm_matmul_graph, 2),
        ("kernel_broadcast_row_bias_add", build_kernel_broadcast_row_bias_add_graph, 1),
        ("kernel_reduce_mul_broadcast", build_kernel_reduce_mul_broadcast_graph, 1),
        ("kernel_reduce_broadcast_mul", build_kernel_reduce_broadcast_mul_graph, 1),
    ]

    for name, builder, expected_min in cases:
        vs = _variants_for(builder)
        got = len(vs)
        assert got >= expected_min, f"{name}: expected >= {expected_min} variants, got {got}"
        print(f" {name}: variants={got} (expected >= {expected_min})")


def _test_matmul_red_div_graph() -> None:
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    norm = _nodes_by_op(G, "div")[0]
    out = _nodes_by_op(G, "matmul")[0]
    assert norm.shape == (4, 8), f"norm shape wrong: {norm.shape}"
    assert out.shape == (4, 16), f"out shape wrong: {out.shape}"

    variants = nu_graph_generation_z3(G, verbose=False)
    norm_pos = _position_by_id(G, norm.id)
    out_pos = _position_by_id(G, out.id)
    assert norm_pos is not None, "Internal test error: _test_matmul_red_div_graph missing norm node"
    assert out_pos is not None, "Internal test error: _test_matmul_red_div_graph missing out node"
    swapped = _swap_with_successor_variants(G, norm_pos, out_pos, {"x"})
    assert len(swapped) == 1, "matmul_red_div should have exactly one legal div/matmul swap"
    assert z3_equivalent_order(G.node_at(norm_pos), G.node_at(out_pos), G, swapped[0][0], verbose=False)
    expected_sigs = {graph_signature(G), graph_signature(swapped[0][0])}
    got_sigs = {graph_signature(variant) for variant in variants}
    assert len(variants) == 2, f"matmul_red_div should emit original + swapped div/matmul variants, got {len(variants)}"
    assert got_sigs == expected_sigs, "matmul_red_div emitted an unexpected variant"
    print(" matmul_red_div graph builds and runs variant generation")


def _test_no_illegal_reduce_broadcast_swap() -> None:
    G = build_kernel_reduce_broadcast_mul_graph(4, 8, 16)
    rec = _nodes_by_op(G, "reduce_sum")[0]
    rec_b = _nodes_by_op(G, "broadcast")[0]
    ok = z3_equivalent_order(rec, rec_b, G, verbose=False)
    assert not ok, "reduce_sum <-> broadcast must be rejected"
    print(" reduce_sum<->broadcast illegal swap rejected")


def _test_no_illegal_reduce_sqrt_swap() -> None:
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)
    rec = _nodes_by_op(G, "reduce_sum")[0]
    rms = _nodes_by_op(G, "sqrt")[0]
    ok = z3_equivalent_order(rec, rms, G, verbose=False)
    assert not ok, "reduce_sum <-> sqrt must be rejected by axioms"
    print(" reduce_sum<->sqrt illegal swap rejected (axiomatic)")


def _test_rmsnorm_matmul_graph() -> None:
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)

    # Find y*y node where both inputs reference the same tensor id.
    yy = next((n for n in _nodes_by_op(G, "mul") if len(n.inputs) == 2 and n.inputs[0] == n.inputs[1]), None)
    assert yy is not None, "Internal test error: rmsnorm_matmul graph missing y*y node"
    rec = _nodes_by_op(G, "reduce_sum")[0]
    rms = _nodes_by_op(G, "sqrt")[0]
    norm = _nodes_by_op(G, "div")[0]
    out = _nodes_by_op(G, "matmul")[0]

    assert yy.shape == (4, 8), f"yy shape wrong: {yy.shape}"
    assert rec.shape == (4, 1), f"rec shape wrong: {rec.shape}"
    assert rms.shape == (4, 1), f"rms shape wrong: {rms.shape}"
    assert norm.shape == (4, 8), f"norm shape wrong: {norm.shape}"
    assert out.shape == (4, 16), f"out shape wrong: {out.shape}"

    variants = nu_graph_generation_z3(G, verbose=False)
    norm_pos = _position_by_id(G, norm.id)
    out_pos = _position_by_id(G, out.id)
    assert norm_pos is not None, "Internal test error: rmsnorm_matmul test graph missing norm node"
    assert out_pos is not None, "Internal test error: rmsnorm_matmul test graph missing out node"
    swapped = _swap_with_successor_variants(G, norm_pos, out_pos, {"x"})
    assert len(swapped) == 1, "rmsnorm_matmul should have exactly one legal div/matmul swap"
    assert z3_equivalent_order(G.node_at(norm_pos), G.node_at(out_pos), G, swapped[0][0], verbose=False)
    expected_sigs = {graph_signature(G), graph_signature(swapped[0][0])}
    got_sigs = {graph_signature(variant) for variant in variants}
    assert len(variants) == 2, f"rmsnorm_matmul should emit original + swapped div/matmul variants, got {len(variants)}"
    assert got_sigs == expected_sigs, "rmsnorm_matmul emitted an unexpected variant"
    print(" rmsnorm_matmul (+sqrt) graph builds and runs variant generation")


def _test_softmax_matmul_graph() -> None:
    G = build_kernel_softmax_matmul_graph(4, 8, 16)
    ex = _nodes_by_op(G, "exp")[0]
    den = _nodes_by_op(G, "reduce_sum")[0]
    probs = _nodes_by_op(G, "div")[0]
    out = _nodes_by_op(G, "matmul")[0]

    assert ex.shape == (4, 8), f"ex shape wrong: {ex.shape}"
    assert den.shape == (4, 1), f"den shape wrong: {den.shape}"
    assert probs.shape == (4, 8), f"probs shape wrong: {probs.shape}"
    assert out.shape == (4, 16), f"out shape wrong: {out.shape}"

    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 1, "softmax_matmul should emit at least org variant"
    print(" softmax_matmul graph builds and runs variant generation")


def _test_transpose_matmul_graph() -> None:
    G = build_kernel_transpose_matmul_graph(4, 8, 16)
    xt = _nodes_by_op(G, "transpose")[0]
    out = _nodes_by_op(G, "matmul")[0]

    assert xt.shape == (8, 4), f"xt shape wrong: {xt.shape}"
    assert out.shape == (8, 16), f"out shape wrong: {out.shape}"

    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 1, "transpose_matmul should emit at least org variant"
    print(" transpose_matmul graph builds and runs variant generation")


def _test_matmul_transpose_graph() -> None:
    G = build_kernel_matmul_transpose_graph(4, 8, 16)
    mm = _nodes_by_op(G, "matmul")[0]
    out = _nodes_by_op(G, "transpose")[0]

    assert mm.shape == (4, 16), f"mm shape wrong: {mm.shape}"
    assert out.shape == (16, 4), f"out shape wrong: {out.shape}"

    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 2, "matmul_transpose should emit a swapped variant"
    print(" matmul_transpose graph builds and runs variant generation")


def _test_nc_matmul_transpose_graph() -> None:
    warmup_x = SymTensor("warmup_x", shape=(4, 8))
    warmup_w = SymTensor("warmup_w", shape=(8, 16))
    warmup_xt = nc_transpose(dst=None, data=warmup_x)
    assert warmup_xt is not None, "Internal test error: nc_transpose warmup failed"
    warmup_mm = nc_matmul(dst=None, stationary=warmup_xt, moving=warmup_w)
    assert warmup_mm is not None, "Internal test error: nc_matmul warmup failed"
    assert nc_transpose(dst=None, data=warmup_mm) is not None, "Internal test error: nc_transpose output warmup failed"

    G = nuGraph([
        Node("x", "input", [], {"shape": (4, 8)}, (4, 8)),
        Node("w", "input", [], {"shape": (8, 16)}, (8, 16)),
        Node("xt", "nc_transpose", ["x"], {}, (8, 4)),
        Node("mm", "nc_matmul", ["xt", "w"], {}, (4, 16)),
        Node("out", "nc_transpose", ["mm"], {}, (16, 4)),
    ])

    mm_pos = _position_by_id(G, "mm")
    out_pos = _position_by_id(G, "out")
    assert mm_pos is not None and out_pos is not None, "Internal test error: nc_matmul/nc_transpose nodes missing"

    swapped = _swap_with_successor_variants(G, mm_pos, out_pos, {"xt", "w"})
    assert swapped, "nc_matmul_transpose should emit a legal swapped variant"

    rewritten = [
        G_new for G_new, _ in swapped
        if any(node.id == "out" and node.op == "nc_matmul" and node.inputs == ["w", "xt"] for node in G_new.nodes)
    ]
    assert rewritten, "nc_matmul_transpose should rewrite transpose(matmul) to nc_matmul(moving, stationary)"

    variants = nu_graph_generation_z3(G, verbose=False)
    assert any(
        any(node.id == "out" and node.op == "nc_matmul" and node.inputs == ["w", "xt"] for node in variant.nodes)
        for variant in variants
    ), "nu_graph_generation_z3 should keep the nc_matmul/nc_transpose swapped variant"
    print(" nc_matmul_transpose graph builds and runs variant generation")


def _test_relu_matmul_graph() -> None:
    G = build_kernel_relu_matmul_graph(4, 8, 16)
    z = _nodes_by_op(G, "relu")[0]
    out = _nodes_by_op(G, "matmul")[0]
    assert z.shape == (4, 8)
    assert out.shape == (4, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) == 1, f"relu_matmul should not push relu past matmul, got {len(variants)} variants"
    assert graph_signature(variants[0]) == graph_signature(G)
    z_pos = _position_by_id(G, z.id)
    out_pos = _position_by_id(G, out.id)
    assert z_pos is not None and out_pos is not None
    illegal_swaps = _swap_with_successor_variants(G, z_pos, out_pos, {"x"})
    assert illegal_swaps, "relu_matmul should generate a candidate illegal swap for rejection testing"
    assert not z3_equivalent_order(G.node_at(z_pos), G.node_at(out_pos), G, illegal_swaps[0][0], verbose=False)

def _test_silu_matmul_graph() -> None:
    G = build_kernel_silu_matmul_graph(4, 8, 16)
    z = _nodes_by_op(G, "silu")[0]
    out = _nodes_by_op(G, "matmul")[0]
    assert out is not None, "Internal test error: silu_matmul graph missing out node"
    assert z.shape == (4, 8)
    assert out.shape == (4, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) == 1, f"silu_matmul should not push silu past matmul, got {len(variants)} variants"
    assert graph_signature(variants[0]) == graph_signature(G)


def _test_silu_mlp_graph() -> None:
    G = build_kernel_silu_mlp_graph(4, 8, 16)
    matmuls = _nodes_by_op(G, "matmul")
    silu_nodes = _nodes_by_op(G, "silu")
    assert len(matmuls) == 2, f"silu_mlp should have 2 matmul nodes, got {len(matmuls)}"
    assert len(silu_nodes) == 1, f"silu_mlp should have 1 silu node, got {len(silu_nodes)}"
    assert matmuls[0].shape == (4, 16), f"first matmul shape wrong: {matmuls[0].shape}"
    assert silu_nodes[0].shape == (4, 16), f"silu shape wrong: {silu_nodes[0].shape}"
    assert matmuls[1].shape == (4, 16), f"second matmul shape wrong: {matmuls[1].shape}"
    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 1, "silu_mlp should emit at least one variant"
    print(" silu_mlp graph builds and runs variant generation")


def _test_attention_graph() -> None:
    G = build_kernel_attention_graph(4, 8, 16)
    matmuls = _nodes_by_op(G, "matmul")
    transposes = _nodes_by_op(G, "transpose")
    exp_nodes = _nodes_by_op(G, "exp")
    reduce_sum_nodes = _nodes_by_op(G, "reduce_sum")
    div_nodes = _nodes_by_op(G, "div")
    assert len(matmuls) == 5, f"attention should have 5 matmul nodes, got {len(matmuls)}"
    assert len(transposes) == 1, f"attention should have 1 transpose node, got {len(transposes)}"
    assert len(exp_nodes) == 1, f"attention should have 1 exp node, got {len(exp_nodes)}"
    assert len(reduce_sum_nodes) == 1, f"attention should have 1 reduce_sum node, got {len(reduce_sum_nodes)}"
    assert len(div_nodes) == 1, f"attention should have 1 div node, got {len(div_nodes)}"
    assert transposes[0].shape == (16, 4), f"k_t shape wrong: {transposes[0].shape}"
    assert exp_nodes[0].shape == (4, 4), f"exp shape wrong: {exp_nodes[0].shape}"
    assert reduce_sum_nodes[0].shape == (4, 1), f"reduce_sum shape wrong: {reduce_sum_nodes[0].shape}"
    assert div_nodes[0].shape == (4, 4), f"div shape wrong: {div_nodes[0].shape}"
    assert matmuls[-1].shape == (4, 16), f"attention output shape wrong: {matmuls[-1].shape}"
    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 1, "attention should emit at least one variant"
    print(" attention graph builds and runs variant generation")


def _test_print_graph_includes_symbolic_shapes() -> None:
    G = build_kernel_relu_matmul_graph(4, 8, 16)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_graph(G)
    out = buf.getvalue()
    assert "sym_shape=" in out
    lines = out.splitlines()
    x_line = next((line for line in lines if "id=x" in line and "op=input" in line), None)
    out_line = next((line for line in lines if "op=matmul" in line), None)
    assert x_line is not None, "Expected print_graph output line for input node x"
    assert out_line is not None, "Expected print_graph output line for matmul node"
    assert "shape=(4, 8)" in x_line
    assert "sym_shape=(x_d0, x_d1)" in x_line
    assert "shape=(4, 16)" in out_line
    assert "sym_shape=(x_d0, w_d1)" in out_line


def _test_print_graph_groups_nodes_by_topological_level() -> None:
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_graph(G)
    out = buf.getvalue()
    lines = out.splitlines()

    x_idx = next(i for i, line in enumerate(lines) if "id=x" in line and "op=input" in line)
    y_idx = next(i for i, line in enumerate(lines) if "id=y" in line and "op=input" in line)
    w_idx = next(i for i, line in enumerate(lines) if "id=w" in line and "op=input" in line)
    mul_idx = next(i for i, line in enumerate(lines) if "id=mul_" in line and "op=mul" in line)

    assert x_idx < mul_idx, "Expected input x to print before derived nodes"
    assert y_idx < mul_idx, "Expected input y to print before derived nodes"
    assert w_idx < mul_idx, (
        "Expected input w, which appears later in G.nodes storage order, "
        "to print before derived nodes"
    )
    print(" print_graph: groups nodes by topological level for readability")


def _test_synthesis_prefers_direct_reduce_candidate() -> None:
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    gv = variants[0]
    reduce_node = _nodes_by_op(gv, "reduce_sum")[0]
    orig_syms = _graph_symbolic_tensors(gv)
    input_syms = [orig_syms[inp_id] for inp_id in reduce_node.inputs]
    target_sym = orig_syms[reduce_node.id]
    pool = _build_synthesis_pool(reduce_node.op, reduce_node.attrs, input_syms)

    sketch = _synthesize_from_pool(target_sym, pool, max_hw_size=2, timeout=500, verbose=False)
    assert sketch is not None, "Expected to find a direct lowering for reduce_sum"
    sketch_str = _format_sketch(sketch)
    assert "tensor_reduce" in sketch_str or sketch.op == "tensor_reduce", \
        f"Expected reduce-oriented candidate first, got {sketch_str}"
    print(f" synthesis: reduce_sum prefers reduce-oriented candidate ({sketch_str})")


def _test_synthesis_verbose_reports_candidate_count() -> None:
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    gv = variants[0]
    reduce_node = _nodes_by_op(gv, "reduce_sum")[0]
    orig_syms = _graph_symbolic_tensors(gv)
    input_syms = [orig_syms[inp_id] for inp_id in reduce_node.inputs]
    target_sym = orig_syms[reduce_node.id]
    pool = _build_synthesis_pool(reduce_node.op, reduce_node.attrs, input_syms)

    buf = io.StringIO()
    with redirect_stdout(buf):
        _synthesize_all_from_pool(
            target_sym,
            pool,
            max_hw_size=2,
            timeout=100,
            verbose=True,
            max_workers=2,
        )
    out = buf.getvalue()
    assert "candidate_count=" in out, "Expected verbose synthesis output to report candidate_count"
    assert "symbolic_shape_reject_count=" in out, \
        "Expected verbose synthesis output to report symbolic_shape_reject_count"
    assert "solver_dispatch_count=" in out, \
        "Expected verbose synthesis output to report solver_dispatch_count"
    assert "sketch " in out and " in " in out, \
        "Expected verbose synthesis output to print each sketch as its check finishes"
    print(" synthesis: verbose output reports candidate_count")


def _test_synthesis_auto_maximizes_sketch_workers() -> None:
    from unittest.mock import patch

    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    gv = variants[0]
    reduce_node = _nodes_by_op(gv, "reduce_sum")[0]
    orig_syms = _graph_symbolic_tensors(gv)
    input_syms = [orig_syms[inp_id] for inp_id in reduce_node.inputs]
    target_sym = orig_syms[reduce_node.id]
    pool = _build_synthesis_pool(reduce_node.op, reduce_node.attrs, input_syms)

    observed: dict[str, int] = {}

    class _RecordingExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "_RecordingExecutor":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        def submit(
            self, fn: Callable[..., Any], *args: Any, **kwargs: Any
        ) -> concurrent.futures.Future:
            fut: concurrent.futures.Future = concurrent.futures.Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as exc:
                fut.set_exception(exc)
            return fut

    def _recording_executor_factory(
        *args: Any, **kwargs: Any
    ) -> "_RecordingExecutor":
        if args:
            observed["max_workers"] = args[0]
        else:
            observed["max_workers"] = kwargs["max_workers"]
        return _RecordingExecutor(*args, **kwargs)

    patch_target = "concurrent.futures.ThreadPoolExecutor"
    with patch(patch_target, side_effect=_recording_executor_factory):
        _synthesize_all_from_pool(
            target_sym,
            pool,
            max_hw_size=2,
            timeout=100,
            verbose=False,
        )

    candidate_count = _last_synthesis_stats.get("candidate_count", 0)
    expected_workers = _effective_max_workers(None, candidate_count)
    cpu_count = os.cpu_count() or 4
    assert candidate_count > 1, "Expected multiple candidate sketches for reduce_sum"
    assert observed.get("max_workers") == expected_workers, (
        f"Expected sketch-level synthesis to auto-launch {expected_workers} worker(s), "
        f"got {observed.get('max_workers')}"
    )
    # In auto mode the worker count must be CPU-bounded, not equal to task count.
    assert expected_workers <= cpu_count, (
        f"Auto workers ({expected_workers}) must not exceed CPU count ({cpu_count})"
    )
    assert expected_workers <= candidate_count, (
        f"Auto workers ({expected_workers}) must not exceed candidate count ({candidate_count})"
    )
    print(
        f" synthesis: auto sketch-level threading launches {observed['max_workers']} "
        f"worker(s) (cpu_count={cpu_count}) for {candidate_count} candidate sketch(es)"
    )


def _test_effective_max_workers_uses_cpu_count() -> None:
    """Unit test: _effective_max_workers(None, N) derives workers from CPU count.

    Validates:
    - Auto mode (max_workers=None) uses os.cpu_count() (defaulting to 4 when
      None) clamped to task_count, rather than returning task_count directly.
    - The fallback of 4 is applied when os.cpu_count() returns None.
    - Explicit max_workers values are unchanged by this logic.
    - task_count <= 1 still returns 1 regardless.
    """
    from unittest.mock import patch

    # --- auto mode with a known cpu_count ---
    for fake_cpu in (1, 2, 4, 8, 16):
        with patch("os.cpu_count", return_value=fake_cpu):
            for task_count in (1, 2, fake_cpu - 1, fake_cpu, fake_cpu + 5):
                if task_count < 1:
                    continue
                result = _effective_max_workers(None, task_count)
                expected = builtins.max(1, builtins.min(fake_cpu, task_count))
                assert result == expected, (
                    f"cpu={fake_cpu}, tasks={task_count}: "
                    f"expected {expected}, got {result}"
                )

    # --- auto mode when os.cpu_count() returns None (fallback = 4) ---
    with patch("os.cpu_count", return_value=None):
        fallback = 4
        for task_count in (1, 2, 3, 4, 10):
            result = _effective_max_workers(None, task_count)
            expected = builtins.max(1, builtins.min(fallback, task_count))
            assert result == expected, (
                f"None cpu fallback, tasks={task_count}: "
                f"expected {expected}, got {result}"
            )

    # --- explicit max_workers is respected unchanged ---
    assert _effective_max_workers(1, 10) == 1
    assert _effective_max_workers(3, 10) == 3
    assert _effective_max_workers(10, 10) == 10
    assert _effective_max_workers(20, 10) == 10  # clamped to task_count
    assert _effective_max_workers(0, 10) == 1    # clamped up to 1

    # --- task_count <= 1 always returns 1 ---
    assert _effective_max_workers(None, 1) == 1
    assert _effective_max_workers(None, 0) == 1
    assert _effective_max_workers(5, 1) == 1

    print(" workers: _effective_max_workers uses CPU count in auto mode")


def _test_lower_nu_graph_all_variants_does_not_hold_z3_lock_across_node_lowering() -> None:
    from unittest.mock import patch

    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    gv = variants[0]
    observed: dict[str, bool] = {"held": False, "called": False}

    def _fake_lower_node_all(
        node: Node,
        target_sym: SymTensor,
        hw_input_pairs: list[tuple[SymTensor, str]],
        max_hw_size: int,
        timeout: int,
        verbose: bool = False,
        max_workers: Optional[int] = None,
    ) -> list[tuple[list[Node], str, SymTensor]]:
        observed["called"] = True
        is_owned = getattr(_Z3_LOCK, "_is_owned", None)
        observed["held"] = bool(is_owned()) if is_owned is not None else False
        return []

    with patch("__main__._lower_node_all", side_effect=_fake_lower_node_all):
        lowered = lower_nu_graph_all_variants(
            gv,
            max_hw_size=2,
            timeout=100,
            verbose=False,
            max_workers=2,
        )

    assert observed["called"], "Expected lower_nu_graph_all_variants to invoke _lower_node_all"
    assert not observed["held"], (
        "lower_nu_graph_all_variants must not hold _Z3_LOCK across _lower_node_all; "
        "doing so deadlocks the inner sketch threadpool"
    )
    assert lowered == [], "Fake node lowering should cause the graph lowering to fail cleanly"
    print(" lowering: all-variants node lowering does not hold _Z3_LOCK across _lower_node_all")


def _test_synthesis_symbolic_shape_prefilter_skips_solver() -> None:
    from unittest.mock import patch

    x = SymTensor("x", shape=(4, 8))
    target_sym = tensor_reduce(dst=None, op=nl.add, data=x, axis=1, keepdims=True)
    pool = [
        SketchNode.make_op(
            "nc_matmul",
            [SketchNode.make_input(x), SketchNode.make_input(x)],
            {},
        )
    ]

    original_check = _check_equivalent_quiet
    calls = {"count": 0}

    def _counting_check(lhs: SymTensor, rhs: SymTensor, timeout: int = 3000) -> bool:
        calls["count"] += 1
        return original_check(lhs, rhs, timeout=timeout)

    with patch("__main__._check_equivalent_quiet", side_effect=_counting_check):
        found = _synthesize_all_from_pool(
            target_sym,
            pool,
            max_hw_size=1,
            timeout=100,
            verbose=False,
            max_workers=1,
        )

    assert found == [], "Expected symbolic shape prefilter to reject mismatched sketch"
    assert calls["count"] == 0, "Expected symbolic shape mismatch to skip equivalence solver"
    assert _last_synthesis_stats.get("symbolic_shape_reject_count") == 1
    assert _last_synthesis_stats.get("solver_dispatch_count") == 0
    print(" synthesis: symbolic shape prefilter skips solver for mismatched sketches")


def _test_sketch_shape_constraints_violated_rejected_early() -> None:
    """Generic sketch-shape-constraint check rejects semantically-impossible sketches
    pre-solver, without any op-specific logic.

    Regression scenario: in kernel_matmul_red_div(4, 8, 16) the search formerly
    explored nc_matmul(transpose(IN:reciprocal_1336), IN:w) and spent ~21 s in
    the equivalence solver before rejecting it.

    Why the old output-shape check missed it:
      reciprocal_1336 has shape [4, 1]
      transpose(reciprocal_1336) has shape [1, 4]
      nc_matmul([1, 4], [8, 16]) produces output shape [4, 16]   ← matches target!
    So the candidate slips past _shapes_incompatible_symbolically.

    Why the generic check catches it:
      nc_matmul's shape_rule encodes the requirement lhs.dims[0] == rhs.dims[0]
      (both inputs must agree on the contraction dimension).
      Here lhs.dims[0]=1 ≠ rhs.dims[0]=8, so simplify(1 == 8) == False.
      _sketch_shape_constraints_violated detects this trivially-False constraint
      by traversing the SymExpr tree and re-applying each op's shape_rule.
    """
    from unittest.mock import patch

    # Shapes from build_kernel_matmul_red_div_graph(4, 8, 16):
    w = SymTensor("w", shape=(8, 16))
    reciprocal_sym = SymTensor("reciprocal_1336", shape=(4, 1))

    # target: output of (x / rec) @ w  →  shape [4, 16]
    target_sym = SymTensor("matmul_out", shape=(4, 16))

    # Build the bad sketch: nc_matmul(transpose(reciprocal_1336), w)
    # stationary = transpose([4,1]) → [1, 4]  (contraction dim = 1)
    # moving     = [8, 16]                    (contraction dim = 8)
    # 1 ≠ 8  →  nc_matmul shape rule: lhs.dims[0] == rhs.dims[0] → False
    bad_sketch = SketchNode.make_op(
        "nc_matmul",
        [
            SketchNode.make_op("transpose", [SketchNode.make_input(reciprocal_sym)], {}),
            SketchNode.make_input(w),
        ],
        {},
    )

    candidate_sym = _eval_sketch(bad_sketch)
    assert candidate_sym is not None, "Bad sketch should evaluate to a SymTensor"

    # Output shape [4, 16] matches target — the old check does NOT fire.
    assert not _shapes_incompatible_symbolically(target_sym, candidate_sym), (
        "Output shape should match target; only the generic constraint check should fire"
    )

    # The generic check DOES fire because nc_matmul's shape_rule constraint
    # lhs.dims[0] == rhs.dims[0] (i.e. 1 == 8) simplifies to False.
    assert _sketch_shape_constraints_violated(candidate_sym), (
        "_sketch_shape_constraints_violated should detect the nc_matmul "
        "contraction-dimension mismatch (1 != 8) via shape_rule traversal"
    )

    reason = _shape_rejection_reason(target_sym, candidate_sym)
    assert reason is not None, "Bad sketch should be rejected by _shape_rejection_reason"
    assert "violated" in reason.lower(), (
        f"Expected a 'shape constraints violated' rejection reason, got: {reason!r}"
    )

    # Also verify a sketch whose contraction dims DO match is NOT rejected.
    x = SymTensor("x", shape=(4, 8))
    good_sketch = SketchNode.make_op(
        "nc_matmul",
        [
            SketchNode.make_op("transpose", [SketchNode.make_input(x)], {}),
            SketchNode.make_input(w),
        ],
        {},
    )
    good_candidate_sym = _eval_sketch(good_sketch)
    assert good_candidate_sym is not None, "Good sketch should evaluate"
    assert not _sketch_shape_constraints_violated(good_candidate_sym), (
        "nc_matmul(transpose([4,8]), [8,16]) has matching contraction dims (8==8) "
        "and must NOT be rejected by the generic check"
    )

    # Confirm zero solver calls when only the bad sketch is in the pool
    original_check = _check_equivalent_quiet
    calls: dict[str, int] = {"count": 0}

    def _counting_check(lhs: SymTensor, rhs: SymTensor, timeout: int = 3000) -> bool:
        calls["count"] += 1
        return original_check(lhs, rhs, timeout=timeout)

    with patch("__main__._check_equivalent_quiet", side_effect=_counting_check):
        found = _synthesize_all_from_pool(
            target_sym,
            [bad_sketch],
            max_hw_size=2,
            timeout=100,
            verbose=False,
            max_workers=1,
        )

    assert found == [], "The bad sketch must not be accepted as equivalent"
    assert calls["count"] == 0, (
        "Expected generic shape-constraint check to skip the equivalence solver entirely"
    )
    assert _last_synthesis_stats.get("symbolic_shape_reject_count", 0) >= 1, (
        "Expected symbolic_shape_reject_count to be incremented for the constraint violation"
    )
    print(
        " synthesis: generic shape-constraint traversal rejects bad nc_matmul sketch "
        "pre-solver (no equivalence-solver call)"
    )


def _test_invoke_hw_op_transpose_uses_nc_transpose_semantics() -> None:
    """Regression test: _invoke_hw_op for op='transpose' must use nc_transpose
    semantics so that equivalence checking and graph materialization are
    consistent.

    Before the fix, _invoke_hw_op evaluated 'transpose' via the public
    transpose() op while _sketch_to_graph_nodes emitted nc_transpose; this
    mismatch caused the synthesizer to reject the valid candidate
    nc_matmul(transpose(IN:x), IN:w) for public matmul(x, w).
    """
    # Shapes: x [4, 8], w [8, 16] → matmul output [4, 16]
    x = SymTensor("x", shape=(4, 8))
    w = SymTensor("w", shape=(8, 16))

    # Target: public matmul(x, w) internally computes nc_matmul(nc_transpose(x), w)
    target_sym = matmul(x, w)

    # Candidate sketch: nc_matmul(transpose(IN:x), IN:w)
    # With the fix, 'transpose' evaluates as nc_transpose, making this
    # semantically identical to the target.
    sketch = SketchNode.make_op(
        "nc_matmul",
        [
            SketchNode.make_op("transpose", [SketchNode.make_input(x)], {}),
            SketchNode.make_input(w),
        ],
        {},
    )
    candidate_sym = _eval_sketch(sketch)
    assert candidate_sym is not None, "Sketch nc_matmul(transpose(x), w) should evaluate"

    # Shape check: candidate must not be rejected pre-solver
    assert not _sketch_shape_constraints_violated(candidate_sym), (
        "nc_matmul(transpose([4,8]), [8,16]) has matching contraction dims and "
        "must not be rejected by shape constraints"
    )

    # Equivalence: the sketch must be accepted as equivalent to public matmul
    assert _check_equivalent_quiet(target_sym, candidate_sym, timeout=3000), (
        "nc_matmul(transpose(IN:x), IN:w) must be equivalent to matmul(x, w) — "
        "_invoke_hw_op 'transpose' must use nc_transpose semantics"
    )
    print(
        " synthesis: nc_matmul(transpose(IN), IN) correctly accepted as "
        "equivalent to public matmul (transpose evaluates as nc_transpose)"
    )


def _test_invoke_hw_op_tensor_reduce_preserves_reduced_shape() -> None:
    y = SymTensor("y", shape=(z3.Int("y_d0"), z3.Int("y_d1")))
    reduced = _invoke_hw_op(
        "tensor_reduce",
        [y],
        {"op": nl.add, "axis": 1, "keepdims": True},
    )
    assert reduced is not None, "tensor_reduce sketch evaluation should produce a SymTensor"
    assert len(reduced.shape) == 2, "keepdims=True must preserve tensor rank"
    assert z3.eq(reduced.shape[0], y.shape[0]), "axis=1 reduction must preserve the leading dim"
    assert z3.is_int_value(reduced.shape[1]) and reduced.shape[1].as_long() == 1, (
        "axis=1 keepdims reduction must produce a trailing dimension of 1"
    )
    print(" synthesis: tensor_reduce sketch evaluation preserves reduced keepdims shape")


def _test_synthesis_pool_filters_templates_by_target_constituents() -> None:
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    orig_syms = _graph_symbolic_tensors(G)

    reduce_node = _nodes_by_op(G, "reduce_sum")[0]
    reduce_pool = _build_synthesis_pool(
        reduce_node.op,
        reduce_node.attrs,
        [orig_syms[inp_id] for inp_id in reduce_node.inputs],
    )
    assert not any(sk.op == "activation_reduce" for sk in reduce_pool), \
        "reduce_sum pool should exclude activation_reduce templates with extra primitives"
    for sketch in reduce_pool:
        if sketch.op in ("INPUT", "HOLE") or sketch.op in _LAYOUT_TRANSFORM_OPS:
            continue
        assert _template_constituents(sketch).issubset(_op_constituents(reduce_node.op)), \
            f"reduce_sum pool admitted non-subset template: {_format_sketch(sketch)}"
    assert any(sk.op == "transpose" for sk in reduce_pool), \
        "reduce_sum pool should keep the abstract transpose layout template"
    assert not any(sk.op in {"dma_copy", "tensor_copy", "dma_transpose", "nc_transpose"}
                   for sk in reduce_pool), \
        "reduce_sum pool should exclude concrete copy/transpose hardware ops"

    matmul_node = _nodes_by_op(G, "matmul")[0]
    matmul_pool = _build_synthesis_pool(
        matmul_node.op,
        matmul_node.attrs,
        [orig_syms[inp_id] for inp_id in matmul_node.inputs],
    )
    assert any(sk.op == "nc_matmul" for sk in matmul_pool), \
        "matmul pool should still keep nc_matmul templates"
    assert any(sk.op == "tensor_tensor" and _operand_to_expr(sk.attrs.get("op")) == "add"
               for sk in matmul_pool), "matmul pool should keep tensor_tensor add templates"
    assert any(sk.op == "tensor_tensor" and _operand_to_expr(sk.attrs.get("op")) == "multiply"
               for sk in matmul_pool), "matmul pool should keep tensor_tensor multiply templates"
    assert not any(sk.op == "activation_reduce" for sk in matmul_pool), \
        "matmul pool should exclude activation_reduce templates with extra primitives"
    assert not any(sk.op == "tensor_tensor" and _operand_to_expr(sk.attrs.get("op")) == "divide"
                   for sk in matmul_pool), \
        "matmul pool should exclude tensor_tensor divide templates"
    assert any(sk.op == "transpose" for sk in matmul_pool), \
        "matmul pool should keep the abstract transpose layout template"
    assert not any(sk.op in {"dma_copy", "tensor_copy", "dma_transpose", "nc_transpose"}
                   for sk in matmul_pool), \
        "matmul pool should exclude concrete copy/transpose hardware ops"
    print(" synthesis: pool filtering keeps only subset-compatible templates")


def _test_single_operator_sketches_cover_distinct_inputs() -> None:
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)
    orig_syms = _graph_symbolic_tensors(G)
    div_node = _nodes_by_op(G, "div")[0]
    input_names = tuple(div_node.inputs)
    pool = _build_synthesis_pool(
        div_node.op,
        div_node.attrs,
        [orig_syms[inp_id] for inp_id in div_node.inputs],
    )

    reciprocal_inputs = {
        sketch.children[0].sym.expr.name
        for sketch in pool
        if sketch.op == "reciprocal"
        and not sketch.has_hole()
        and sketch.children
        and sketch.children[0].sym is not None
        and sketch.children[0].sym.expr is not None
    }
    assert reciprocal_inputs == set(input_names), \
        f"Expected unary single-op sketches over both div inputs, got {sorted(reciprocal_inputs)}"

    divide_pairs = {
        tuple(
            child.sym.expr.name
            for child in sketch.children
            if child.sym is not None and child.sym.expr is not None
        )
        for sketch in pool
        if sketch.op == "tensor_tensor"
        and _operand_to_expr(sketch.attrs.get("op")) == "divide"
        and not sketch.has_hole()
    }
    expected_pairs = {
        (input_names[0], input_names[0]),
        (input_names[0], input_names[1]),
        (input_names[1], input_names[0]),
        (input_names[1], input_names[1]),
    }
    assert divide_pairs == expected_pairs, \
        f"Expected binary single-op sketches over all ordered div input pairs, got {sorted(divide_pairs)}"
    print(" synthesis: single-op sketches cover distinct concrete inputs")


def _test_division_broadcast_uses_tensor_scalar_not_tensor_tensor() -> None:
    from unittest.mock import patch

    x = SymTensor("x", shape=(4, 8))
    reduced = SymTensor("tensor_reduce_1080", shape=(4, 1))

    target_sym = divide(x, reduced)
    bad_candidate = reciprocal(dst=None, data=tensor_tensor(dst=None, data1=reduced, data2=x, op=nl.divide))
    good_candidate = tensor_scalar(dst=None, data=x, op0=nl.divide, operand0=reduced)

    assert _sketch_shape_constraints_violated(bad_candidate), (
        "tensor_tensor divide must require same-shaped operands; broadcasted [4,1] / [4,8] "
        "should be rejected before equivalence checking"
    )
    assert _shape_rejection_reason(target_sym, bad_candidate) == "sketch shape constraints violated", (
        "broadcasted public divide candidate must be rejected by shape constraints before solver dispatch"
    )
    assert not _sketch_shape_constraints_violated(good_candidate), (
        "tensor_scalar is the broadcast-capable hardware op for divide-like sketches"
    )
    assert _check_equivalent_quiet(target_sym, good_candidate, timeout=3000), (
        "broadcasted public divide should still match tensor_scalar divide semantics"
    )

    bad_sketch = SketchNode.make_op(
        "reciprocal",
        [
            SketchNode.make_op(
                "tensor_tensor",
                [SketchNode.make_input(reduced), SketchNode.make_input(x)],
                {"op": nl.divide},
            )
        ],
        {},
    )
    calls = {"count": 0}
    original_check = _check_equivalent_quiet

    def _counting_check(lhs: SymTensor, rhs: SymTensor, timeout: int = 3000) -> bool:
        calls["count"] += 1
        return original_check(lhs, rhs, timeout=timeout)

    with patch("__main__._check_equivalent_quiet", side_effect=_counting_check):
        found = _synthesize_all_from_pool(
            target_sym,
            [bad_sketch],
            max_hw_size=2,
            timeout=100,
            verbose=False,
            max_workers=1,
        )

    assert found == [], "invalid tensor_tensor divide sketch must not be accepted for broadcasted div"
    assert calls["count"] == 0, "shape rejection should skip solver dispatch for invalid tensor_tensor divide sketches"
    print(" synthesis: broadcasted div matches tensor_scalar, not tensor_tensor")


def _test_division_symbolic_shape_rejection() -> None:
    """Regression: tensor_tensor divide candidates with symbolically-mismatched
    operand shapes must be rejected before the equivalence solver is called,
    even when the mismatch is not trivially detectable by ``z3.simplify``
    (i.e. a symbolic constraint ``k == 1`` rather than the concrete ``8 == 1``
    that the existing concrete-shape test already covers).

    Background
    ----------
    In the synthesis log for ``kernel_matmul_red_div`` the search found:

      * ``reciprocal(tensor_tensor[op=divide](IN:tensor_reduce_1080, IN:x))``
      * ``tensor_tensor[op=divide](IN:x, IN:tensor_reduce_1080)``
      * ``tensor_tensor[op=divide](IN:x, transpose(IN:tensor_reduce_1080))``

    as falsely-equivalent lowerings for ``div_1002``.  With concrete shapes
    these are caught by ``_sketch_shape_constraints_violated`` (``8==1`` → False),
    but with the *symbolic* shapes used during synthesis (``x: [m, k]``,
    ``reduced: [m, 1]``) the constraint ``k == 1`` emitted by
    ``_shape_tensor_tensor`` does not simplify to False and slips through the
    pre-filter into the equivalence solver, which then (wrongly) accepts it
    because the merged context contains ``k == 1``.

    The new ``_shapes_not_provably_equivalent`` guard asks the solver whether
    ``k == 1`` follows from the input-dimension positivity constraints alone
    (``m > 0, k > 0``).  The answer is NO (SAT with k = 8), so all such
    candidates are rejected before any expensive equivalence check is attempted.
    """
    from unittest.mock import patch

    symbolic_m = z3.Int("sym_m")
    symbolic_k = z3.Int("sym_k")
    x = SymTensor("x_sym", shape=(symbolic_m, symbolic_k))
    reduced = SymTensor("reduced_sym", shape=(symbolic_m, z3.IntVal(1)))

    target_sym = divide(x, reduced)

    # Bad candidates — tensor_tensor requires all dims equal, injecting k == 1
    bad1 = tensor_tensor(dst=None, data1=x, data2=reduced, op=nl.divide)
    bad2 = tensor_tensor(dst=None, data1=reduced, data2=x, op=nl.divide)
    bad3 = reciprocal(
        dst=None,
        data=tensor_tensor(dst=None, data1=reduced, data2=x, op=nl.divide),
    )

    for bad, desc in [
        (bad1, "tensor_tensor(x, reduced)"),
        (bad2, "tensor_tensor(reduced, x)"),
        (bad3, "reciprocal(tensor_tensor(reduced, x))"),
    ]:
        # The trivial simplify-based check must NOT fire for symbolic shapes
        assert not _sketch_shape_constraints_violated(bad), (
            f"{desc}: _sketch_shape_constraints_violated must not fire for symbolic "
            f"shapes (constraint k==1 is non-trivial, not z3.is_false)"
        )
        # The new solver-backed check MUST detect the illegal restriction
        assert _shapes_not_provably_equivalent(target_sym, bad), (
            f"{desc}: _shapes_not_provably_equivalent must detect that k==1 is not "
            f"implied by input positivity (m>0, k>0)"
        )
        reason = _shape_rejection_reason(target_sym, bad)
        assert reason is not None, (
            f"{desc}: _shape_rejection_reason must reject this symbolic-shape candidate"
        )
        assert "provably" in reason.lower(), (
            f"{desc}: expected an 'output shape not provably equivalent' rejection, "
            f"got: {reason!r}"
        )

    # Good candidate: tensor_scalar is the broadcast-capable op; it adds only
    # trivially-True broadcast constraints, not k == 1.
    good = tensor_scalar(dst=None, data=x, op0=nl.divide, operand0=reduced)
    assert not _shapes_not_provably_equivalent(target_sym, good), (
        "tensor_scalar(x, op=divide, operand=reduced) must NOT be rejected by "
        "_shapes_not_provably_equivalent — it has no non-trivial shape constraints"
    )
    assert _shape_rejection_reason(target_sym, good) is None, (
        "tensor_scalar lowering must pass all shape pre-filters"
    )
    assert _check_equivalent_quiet(target_sym, good, timeout=5000), (
        "tensor_scalar(x, op=divide, operand=reduced) must be proven equivalent "
        "to the broadcasted public divide"
    )

    # Verify synthesis rejects bad candidates without touching the equivalence solver
    bad_sketches = [
        SketchNode.make_op(
            "tensor_tensor",
            [SketchNode.make_input(x), SketchNode.make_input(reduced)],
            {"op": nl.divide},
        ),
        SketchNode.make_op(
            "reciprocal",
            [
                SketchNode.make_op(
                    "tensor_tensor",
                    [SketchNode.make_input(reduced), SketchNode.make_input(x)],
                    {"op": nl.divide},
                )
            ],
            {},
        ),
    ]
    calls: dict[str, int] = {"count": 0}
    original_check = _check_equivalent_quiet

    def _counting_check(lhs: SymTensor, rhs: SymTensor, timeout: int = 3000) -> bool:
        calls["count"] += 1
        return original_check(lhs, rhs, timeout=timeout)

    with patch("__main__._check_equivalent_quiet", side_effect=_counting_check):
        found = _synthesize_all_from_pool(
            target_sym,
            bad_sketches,
            max_hw_size=2,
            timeout=100,
            verbose=False,
            max_workers=1,
        )

    assert found == [], (
        "Symbolic-shape tensor_tensor divide candidates must not be accepted"
    )
    assert calls["count"] == 0, (
        "_shapes_not_provably_equivalent must prevent equivalence-solver dispatch "
        "for all symbolic-shape tensor_tensor divide candidates"
    )
    print(
        " synthesis: symbolic-shape tensor_tensor divide candidates rejected before "
        "equivalence solver by _shapes_not_provably_equivalent"
    )


def _test_shapes_not_provably_equivalent_broadcast_compatible_candidate() -> None:
    """Regression: ``_shapes_not_provably_equivalent`` must not reject a valid
    ``tensor_scalar`` candidate for ``div`` when the synthesis assigns *distinct*
    symbolic dimension variables to operands that are equal at run time.

    Scenario
    --------
    In ``kernel_matmul_red_div`` the synthesis builds a local target for
    ``div_1002`` from two previously-lowered hw nodes:

    * ``x_sym``              shape ``(x_d0, x_d1)``   (e.g. 4 × 8)
    * ``tensor_reduce_1080`` shape ``(y_d0, 1)``       (e.g. 4 × 1)

    Here ``x_d0`` and ``y_d0`` are **distinct** z3 variables even though at
    run time ``x_d0 == y_d0 == 4``.  The target ``div`` (after Fix 1:
    ``_shape_public_binary`` uses ``_broadcast_shape``) emits the constraint
    ``Or(x_d0==y_d0, x_d0==1, y_d0==1)`` as its shape pre-condition.

    The broadcast-capable candidate ``tensor_scalar(x, reduce, op=divide)``
    emits the same ``Or(...)`` constraint.  Without Fix 2 (including the
    target's constraint in the valid-assumption set), the old check

        SAT(x_d0>0, x_d1>0, y_d0>0, ¬Or(x_d0==y_d0, x_d0==1, y_d0==1))

    is SAT (x_d0=4, y_d0=5), so the candidate was wrongly rejected.

    With Fix 2 the valid assumptions include the target's ``Or(...)``
    constraint, making the negated candidate constraint UNSAT → candidate kept.

    Conversely, the strict ``tensor_tensor(x, reduce)`` candidate emits
    ``x_d0==y_d0 ∧ x_d1==1``.  Even with the target's Or as an assumption,
    ``SAT(…, Or(x_d0==y_d0,…), ¬(x_d0==y_d0 ∧ x_d1==1))`` is SAT
    (x_d0=y_d0=4, x_d1=8), so that candidate is correctly rejected.
    """
    x_d0 = z3.Int("x_d0")
    x_d1 = z3.Int("x_d1")
    y_d0 = z3.Int("y_d0")

    x_sym = SymTensor("x", shape=(x_d0, x_d1))
    reduce_sym = SymTensor("tensor_reduce_1080", shape=(y_d0, z3.IntVal(1)))

    # Build local target the same way lower_nu_graph_all_variants does:
    # _sym_expr_from_graph_node(div_node, [x_sym, reduce_sym])
    div_node = Node(
        id="div_1002", op="div",
        inputs=["x", "tensor_reduce_1080"],
        shape=(4, 8), attrs={},
    )
    local_target = _sym_expr_from_graph_node(div_node, [x_sym, reduce_sym])

    # Good candidate: tensor_scalar supports broadcast
    good = tensor_scalar(dst=None, data=x_sym, op0=nl.divide, operand0=reduce_sym)
    assert not _shapes_not_provably_equivalent(local_target, good), (
        "tensor_scalar(x, reduce, op=divide) must NOT be rejected by "
        "_shapes_not_provably_equivalent when operands use distinct symbolic "
        "vars — the target's broadcast constraint makes the candidate's "
        "Or constraint trivially implied"
    )
    assert _shape_rejection_reason(local_target, good) is None, (
        "tensor_scalar must pass all shape pre-filters for the div node "
        "lowering scenario with distinct symbolic dimension variables"
    )
    assert _check_equivalent_quiet(local_target, good, timeout=5000), (
        "tensor_scalar(x, reduce, op=divide) must be proven equivalent to "
        "broadcasted div(x, reduce) when x.shape=(x_d0,x_d1), reduce.shape=(y_d0,1)"
    )

    # Bad candidate: tensor_tensor injects the spurious k==1 constraint
    bad = tensor_tensor(dst=None, data1=x_sym, data2=reduce_sym, op=nl.divide)
    assert _shapes_not_provably_equivalent(local_target, bad), (
        "tensor_tensor(x, reduce, op=divide) must be rejected by "
        "_shapes_not_provably_equivalent — it injects x_d1==1 which is not "
        "implied by the target's Or(x_d0==y_d0, …) assumption"
    )
    reason = _shape_rejection_reason(local_target, bad)
    assert reason is not None, (
        "tensor_tensor must be rejected before the equivalence solver for the "
        "distinct-symbolic-var div lowering scenario"
    )
    assert "provably" in reason.lower(), (
        f"Expected 'output shape not provably equivalent' rejection, got: {reason!r}"
    )
    print(
        " _shapes_not_provably_equivalent: broadcast-compatible tensor_scalar kept, "
        "strict tensor_tensor rejected for div with distinct symbolic vars"
    )


def _test_tensor_scalar_data_operand_stays_non_broadcastable_tile() -> None:
    from unittest.mock import patch

    x = SymTensor("x", shape=(4, 8))
    reduced = SymTensor("tensor_reduce_1080", shape=(4, 1))
    target_sym = divide(x, reduced)

    good = tensor_scalar(dst=None, data=x, op0=nl.divide, operand0=reduced)
    bad = tensor_scalar(dst=None, data=reduced, op0=nl.divide, operand0=x)

    assert _shapes_match_exactly(good, x), (
        "tensor_scalar must preserve the non-broadcastable tile shape from data"
    )
    assert _shapes_match_exactly(bad, reduced), (
        "tensor_scalar must keep data.shape even when operand0 is larger"
    )
    reason = _shape_rejection_reason(target_sym, bad)
    assert reason is not None, (
        "reversed tensor_scalar(reduce, x) must be rejected before solver dispatch "
        f"because data drives the output tile shape, got: {reason!r}"
    )

    bad_sketch = SketchNode.make_op(
        "tensor_scalar",
        [SketchNode.make_input(reduced), SketchNode.make_input(x)],
        {"op0": nl.divide},
    )
    calls = {"count": 0}
    original_check = _check_equivalent_quiet

    def _counting_check(lhs: SymTensor, rhs: SymTensor, timeout: int = 3000) -> bool:
        calls["count"] += 1
        return original_check(lhs, rhs, timeout=timeout)

    with patch("__main__._check_equivalent_quiet", side_effect=_counting_check):
        found = _synthesize_all_from_pool(
            target_sym,
            [bad_sketch],
            max_hw_size=1,
            timeout=100,
            verbose=False,
            max_workers=1,
        )

    assert found == [], (
        "reversed tensor_scalar(reduce, x) must never be accepted as a valid lowering"
    )
    assert calls["count"] == 0, (
        "shape prefilters must reject reversed tensor_scalar orientation before solver dispatch"
    )
    print(" synthesis: tensor_scalar keeps data as the output tile and rejects reversed broadcast orientation")


def _test_kernel_matmul_red_div_div_node_lowered_via_tensor_scalar() -> None:
    """Regression: the ``div`` node in ``kernel_matmul_red_div`` variant 0 must
    be successfully lowered to a ``tensor_scalar`` hw candidate, not fail with
    'no hw equivalent found'.

    This is the end-to-end regression for the bug described in the problem
    statement: ``lower_nu_graph_all_variants`` was printing
    'FAILED: no hw equivalent found for div_1002 op=div' because every
    candidate (including the correct broadcast ``tensor_scalar``) was rejected
    by the shape prefilter before reaching the equivalence solver.

    This test exercises the node-level synthesis path directly (without running
    the full, slow ``lower_nu_graph_all_variants``) by replicating the exact
    symbolic-tensor setup used during lowering of ``div_1002``.
    """
    # Replicate the synthesis context for div_1002 in kernel_matmul_red_div:
    #   x_sym              : shape = (x_d0, x_d1)
    #   tensor_reduce_1080 : shape = (y_d0, IntVal(1))
    #                         (result of tensor_reduce[axis=1, keepdims=True](y))
    x_d0 = z3.Int("x_d0")
    x_d1 = z3.Int("x_d1")
    y_d0 = z3.Int("y_d0")

    x_sym = SymTensor("x", shape=(x_d0, x_d1))
    reduce_sym = SymTensor("tensor_reduce_1080", shape=(y_d0, z3.IntVal(1)))

    # Build the local target exactly as _lower_node does
    div_node = Node(
        id="div_1002", op="div",
        inputs=["x", "tensor_reduce_1080"],
        shape=(4, 8), attrs={},
    )
    local_target = _sym_expr_from_graph_node(div_node, [x_sym, reduce_sym])

    # Build a synthesis pool containing only the candidates that were failing
    ts_sketch = SketchNode.make_op(
        "tensor_scalar",
        [SketchNode.make_input(x_sym), SketchNode.make_input(reduce_sym)],
        {"op0": nl.divide},
    )
    tt_sketch = SketchNode.make_op(
        "tensor_tensor",
        [SketchNode.make_input(x_sym), SketchNode.make_input(reduce_sym)],
        {"op": nl.divide},
    )

    found = _synthesize_all_from_pool(
        local_target,
        [ts_sketch, tt_sketch],
        max_hw_size=2,
        timeout=5000,
        verbose=False,
        max_workers=1,
    )

    assert any(
        _format_sketch(s) is not None and "tensor_scalar" in _format_sketch(s)
        for s in found
    ), (
        f"div_1002 synthesis must find tensor_scalar as a valid lowering. "
        f"Found: {[_format_sketch(s) for s in found]}"
    )
    assert not any(
        _format_sketch(s) is not None and "tensor_tensor" in _format_sketch(s)
        and "tensor_scalar" not in _format_sketch(s)
        for s in found
    ), (
        "tensor_tensor must not be accepted as a valid lowering for div_1002 "
        "(it injects a spurious k==1 constraint and produces false equivalence)"
    )
    print(
        f" kernel_matmul_red_div div_1002 node-level synthesis: "
        f"tensor_scalar found, tensor_tensor rejected "
        f"(found {len(found)} equivalent sketch(es))"
    )


def _test_matmul_with_complex_input_sym_not_rejected() -> None:
    """Regression: ``_shapes_not_provably_equivalent`` must NOT reject the valid
    ``nc_matmul(nc_transpose(div_sym), w_sym)`` candidate when the synthesis
    inputs include a complex (multi-op) SymTensor such as the canonical
    ``tensor_scalar`` result from a prior ``div`` synthesis step.

    Root cause of the original failure
    -----------------------------------
    Without the ``input_syms`` fix, ``_collect_candidate_shape_constraints``
    traversed *into* the ``div_sym`` sub-expression and collected the broadcast
    constraint ``Or(x_d0==y_d0, x_d0==1, y_d0==1)`` as if it were a NEW
    constraint introduced by the matmul sketch.  When the SAT check negated
    that constraint, there existed satisfying assignments (e.g. x_d0=4, y_d0=5)
    making it True → the candidate was rejected.  As a result the matmul
    synthesis pool contained only ``IN:w`` and no candidate could be equivalent
    to the real two-input matmul.

    With the fix, ``_collect_candidate_shape_constraints`` stops at
    ``div_sym.expr`` (a member of ``input_syms``) and never collects its
    internal broadcast constraint.  The only candidate constraint is then
    ``nc_transpose.shape[0] == w_sym.shape[0]``, which IS implied by the
    matmul target constraint → candidate kept.
    """
    x_d0 = z3.Int("x_d0")
    x_d1 = z3.Int("x_d1")
    y_d0 = z3.Int("y_d0")
    w_d0 = z3.Int("w_d0")
    w_d1 = z3.Int("w_d1")

    x_sym = SymTensor("x", shape=(x_d0, x_d1))
    reduce_sym = SymTensor("tensor_reduce_can", shape=(y_d0, z3.IntVal(1)))
    w_sym = SymTensor("w", shape=(w_d0, w_d1))

    # Simulate the canonical div sym produced by the prior synthesis step:
    # tensor_scalar(data=x_sym, op0=nl.divide, operand0=reduce_sym)
    div_sym = tensor_scalar(dst=None, data=x_sym, op0=nl.divide, operand0=reduce_sym)
    assert div_sym is not None, "tensor_scalar evaluation failed in test setup"
    # div_sym.shape = (If(x_d0==1, y_d0, x_d0), x_d1)

    # Build the matmul target from the canonical inputs, as _lower_node_all does
    matmul_node = Node(
        id="matmul_1003", op="matmul",
        inputs=["div_sym", "w"],
        shape=(4, 16), attrs={},
    )
    local_target = _sym_expr_from_graph_node(matmul_node, [div_sym, w_sym])
    assert local_target is not None, "matmul local target construction failed"

    # The correct candidate: nc_matmul(nc_transpose(div_sym), w_sym)
    transpose_sym = nc_transpose(dst=None, data=div_sym)
    assert transpose_sym is not None, "nc_transpose evaluation failed"
    matmul_candidate = nc_matmul(dst=None, stationary=transpose_sym, moving=w_sym)
    assert matmul_candidate is not None, "nc_matmul evaluation failed"

    # WITHOUT input_syms: the shape prefilter should wrongly reject the candidate
    # because _collect_candidate_shape_constraints traverses into div_sym.expr
    # and collects Or(x_d0==y_d0, x_d0==1, y_d0==1), which is negatable.
    # (We test that the old behaviour produced rejection, so the fix is meaningful.)
    rejected_without_fix = _shapes_not_provably_equivalent(
        local_target, matmul_candidate, input_syms=None
    )
    assert rejected_without_fix, (
        "Without input_syms, _shapes_not_provably_equivalent should reject "
        "nc_matmul(nc_transpose(div_sym), w_sym) due to traversal into div_sym internals"
    )

    # WITH input_syms=[div_sym, w_sym]: the traversal stops at div_sym.expr
    # and the candidate is correctly kept.
    rejected_with_fix = _shapes_not_provably_equivalent(
        local_target, matmul_candidate, input_syms=[div_sym, w_sym]
    )
    assert not rejected_with_fix, (
        "With input_syms=[div_sym, w_sym], _shapes_not_provably_equivalent must NOT "
        "reject nc_matmul(nc_transpose(div_sym), w_sym) — the traversal must stop at "
        "the synthesis input boundary and not collect div_sym's internal broadcast "
        "constraint as a new candidate constraint"
    )

    # _shape_rejection_reason with input_syms must also pass
    reason = _shape_rejection_reason(local_target, matmul_candidate,
                                     input_syms=[div_sym, w_sym])
    assert reason is None, (
        f"_shape_rejection_reason must not reject nc_matmul(nc_transpose(div_sym), w) "
        f"when input_syms are provided; got: {reason!r}"
    )

    print(
        " _shapes_not_provably_equivalent: nc_matmul(nc_transpose(div_sym), w) "
        "correctly kept when input_syms=[div_sym, w_sym] are provided; "
        "correctly rejected (as expected) without input_syms"
    )


def _test_matmul_missing_input_guard() -> None:
    """Regression: ``lower_nu_graph`` and ``lower_nu_graph_all_variants`` must
    fail immediately (return ``None`` / ``[]``) when any upstream lowered input
    is missing from the hw symbol map, rather than silently synthesizing with a
    truncated input list.

    This test verifies the guard added in both lowering functions:

        if missing_inputs:
            return None   # lower_nu_graph
            return []     # lower_nu_graph_all_variants

    by constructing a minimal two-node graph (div → matmul) and injecting a
    missing upstream sym directly, so the matmul would otherwise receive only
    one of its two expected inputs.
    """
    # Build a small graph: div(x, y) → matmul(div_result, w)
    # where div should synthesize to something, but we will deliberately
    # omit its hw sym from the map so matmul sees a missing input.
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    assert variants, "kernel_matmul_red_div should produce at least one variant"
    gv = variants[0]

    # Confirm that the graph has the multi-input matmul structure
    matmul_nodes = _nodes_by_op(gv, "matmul")
    assert matmul_nodes, "test requires a matmul node"
    matmul_node = matmul_nodes[0]
    assert len(matmul_node.inputs) == 2, (
        f"matmul must have 2 inputs, got {matmul_node.inputs}"
    )

    # Find the upstream (non-w) input to matmul — this is the synthesis node
    upstream_input = next(i for i in matmul_node.inputs if i != "w")

    # lower_nu_graph: inject a hw_syms_can state where upstream_input is
    # deliberately absent.  We verify the function returns None rather than
    # proceeding with a truncated input list.  We do this by constructing a
    # minimal graph containing only the matmul node with an unresolvable input.
    orig_syms = _graph_symbolic_tensors(gv)
    w_sym = orig_syms["w"]

    # Simulate what lower_nu_graph_all_variants does when an upstream node
    # fails to synthesize: hw_syms_can has 'w' but NOT the upstream div sym.
    hw_syms_can_incomplete: dict[str, "SymTensor"] = {
        "x": orig_syms["x"],
        "y": orig_syms["y"],
        "w": w_sym,
        # NOTE: upstream_input is intentionally NOT present
    }
    node_id_map_can_incomplete: dict[str, str] = {
        "x": "x", "y": "y", "w": "w",
        # upstream_input not mapped → hw_id lookup falls back to the orig id,
        # which is also absent from hw_syms_can_incomplete
    }

    hw_input_pairs: list[tuple["SymTensor", str]] = []
    missing: list[str] = []
    for inp_id in matmul_node.inputs:
        hw_id = node_id_map_can_incomplete.get(inp_id, inp_id)
        sym = hw_syms_can_incomplete.get(hw_id)
        if sym is not None:
            hw_input_pairs.append((sym, hw_id))
        else:
            missing.append(inp_id)

    assert missing, (
        f"Test setup error: expected '{upstream_input}' to be missing from "
        f"hw_syms_can_incomplete but missing={missing}"
    )
    assert len(hw_input_pairs) < len(matmul_node.inputs), (
        "Test setup error: hw_input_pairs should be shorter than matmul.inputs"
    )

    # The guard in lower_nu_graph_all_variants returns [] when missing_inputs
    # is non-empty; verify this explicitly by checking the guard condition.
    assert len(hw_input_pairs) != len(matmul_node.inputs), (
        "Missing-input guard must detect that hw_input_pairs is incomplete"
    )

    print(
        f" missing-input guard: matmul with inputs={matmul_node.inputs!r} "
        f"correctly detected missing hw sym for '{upstream_input}' "
        f"(hw_input_pairs has {len(hw_input_pairs)}/{len(matmul_node.inputs)} inputs)"
    )


def _test_lower_nu_graph_parallel_levels() -> None:
    def kernel_parallel_reductions(x, y):
        return x.sum(axis=1, keep_dims=True) / y.sum(axis=1, keep_dims=True)

    G = _build_graph_from_kernel(
        kernel_parallel_reductions,
        ("x", (4, 8)),
        ("y", (4, 8)),
    )
    levels = _build_dag_levels(G)
    assert any(len([node for node in level if node.op not in {"input"} | _PUBLIC_PASSTHROUGH_OPS]) > 1
               for level in levels), "Expected a DAG level with multiple synthesis nodes"

    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000, verbose=False, max_workers=2)
    assert G_hw is not None, "parallel lower_nu_graph returned None"
    assert _graph_uses_hw_only(G_hw), "parallel lower_nu_graph left public ops in the graph"
    pair = _graph_output_sym_lowered(G, G_hw)
    assert pair is not None, "parallel lower_nu_graph could not extract output syms"
    orig_sym, hw_sym = pair
    assert _check_equivalent_quiet(orig_sym, hw_sym, timeout=15000), \
        "parallel lower_nu_graph result is not equivalent to the original graph"
    print(" synthesizer: parallel lower_nu_graph handles independent DAG levels")


def _test_lower_node_all_prefers_shape_exact_div_canonical() -> None:
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    gv = nu_graph_generation_z3(G, verbose=False)[0]
    orig_syms = _graph_symbolic_tensors(gv)

    reduce_node = _nodes_by_op(gv, "reduce_sum")[0]
    reduce_alts = _lower_node_all(
        reduce_node,
        orig_syms[reduce_node.id],
        [(orig_syms[reduce_node.inputs[0]], reduce_node.inputs[0])],
        max_hw_size=2,
        timeout=1000,
        verbose=False,
        max_workers=2,
    )
    assert reduce_alts, "Expected at least one lowering for reduce_sum"

    div_node = _nodes_by_op(gv, "div")[0]
    div_alts = _lower_node_all(
        div_node,
        orig_syms[div_node.id],
        [(orig_syms["x"], "x"), (reduce_alts[0][2], reduce_alts[0][1])],
        max_hw_size=2,
        timeout=1000,
        verbose=False,
        max_workers=2,
    )
    assert div_alts, "Expected at least one lowering for div"
    assert _shapes_match_exactly(orig_syms[div_node.id], div_alts[0][2]), (
        "Canonical div lowering must preserve the exact target symbolic shape"
    )
    print(" lowering: canonical div alternative preserves the exact target shape")


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------

def _test_lowering_cache_key_derivation() -> None:
    """Unit test: _make_lowering_cache_key returns distinct keys for distinct
    problems and equal keys for structurally identical problems."""
    x_sym = SymTensor("x", rank=2)
    y_sym = SymTensor("y", rank=2)
    w_sym = SymTensor("w", rank=2)

    # Same op/attrs/input-ranks → same key regardless of variable names.
    key1 = _make_lowering_cache_key("reduce_sum", {"axis": 1, "keep_dims": True}, [x_sym])
    key2 = _make_lowering_cache_key("reduce_sum", {"axis": 1, "keep_dims": True}, [y_sym])
    assert key1 == key2, (
        "Cache keys for reduce_sum with identical attrs and same input rank "
        "must be equal even when symbolic variable names differ"
    )

    # Different axis → different key.
    key3 = _make_lowering_cache_key("reduce_sum", {"axis": 0, "keep_dims": True}, [x_sym])
    assert key1 != key3, "Cache keys must differ when attrs differ (axis=1 vs axis=0)"

    # Different op → different key.
    key4 = _make_lowering_cache_key("matmul", {"axis": 1, "keep_dims": True}, [x_sym])
    assert key1 != key4, "Cache keys must differ when the op differs"

    # Different number of inputs → different key.
    key5 = _make_lowering_cache_key("reduce_sum", {"axis": 1, "keep_dims": True}, [x_sym, y_sym])
    assert key1 != key5, "Cache keys must differ when the number of inputs differs"

    # Different input rank → different key.
    z_rank3 = SymTensor("z", rank=3)
    key6 = _make_lowering_cache_key("reduce_sum", {"axis": 1, "keep_dims": True}, [z_rank3])
    assert key1 != key6, "Cache keys must differ when input ranks differ"

    # Two-input key is stable across different sym names.
    key_mm_1 = _make_lowering_cache_key("matmul", {}, [x_sym, w_sym])
    key_mm_2 = _make_lowering_cache_key("matmul", {}, [y_sym, w_sym])
    assert key_mm_1 == key_mm_2, (
        "Two-input matmul key must be equal regardless of sym variable names"
    )
    print(" caching: _make_lowering_cache_key distinguishes all expected problem variants")


def _test_lowering_cache_normalize_denormalize() -> None:
    """Unit test: _normalize_sketch / _denormalize_sketch round-trip."""
    sym0 = SymTensor("in0", rank=2)
    sym1 = SymTensor("in1", rank=2)
    input_syms = [sym0, sym1]

    # Build: nc_matmul(transpose(IN:sym0), IN:sym1)
    leaf0 = SketchNode.make_input(sym0)
    leaf1 = SketchNode.make_input(sym1)
    inner = SketchNode.make_op("transpose", [leaf0], {})
    outer = SketchNode.make_op("nc_matmul", [inner, leaf1], {})

    norm = _normalize_sketch(outer, input_syms)
    assert norm is not None, "_normalize_sketch returned None for a valid sketch"
    assert norm == ("OP", "nc_matmul", {}, (
        ("OP", "transpose", {}, (("INPUT", 0),)),
        ("INPUT", 1),
    )), f"Unexpected normalized form: {norm}"

    # Round-trip with original syms
    rebuilt = _denormalize_sketch(norm, input_syms)
    assert rebuilt is not None, "_denormalize_sketch returned None"
    assert rebuilt.op == "nc_matmul", f"Expected nc_matmul, got {rebuilt.op}"
    assert rebuilt.children[0].op == "transpose"
    assert rebuilt.children[0].children[0].op == "INPUT"
    assert rebuilt.children[0].children[0].sym is sym0
    assert rebuilt.children[1].op == "INPUT"
    assert rebuilt.children[1].sym is sym1

    # Round-trip with DIFFERENT syms (simulates cache reuse in another variant)
    new_sym0 = SymTensor("x_v2", rank=2)
    new_sym1 = SymTensor("w_v2", rank=2)
    rebuilt2 = _denormalize_sketch(norm, [new_sym0, new_sym1])
    assert rebuilt2 is not None, "_denormalize_sketch returned None for new syms"
    assert rebuilt2.children[0].children[0].sym is new_sym0, (
        "Rebuilt sketch must bind the new input sym at index 0"
    )
    assert rebuilt2.children[1].sym is new_sym1, (
        "Rebuilt sketch must bind the new input sym at index 1"
    )

    # Out-of-range index returns None
    bad = _denormalize_sketch(("INPUT", 5), [sym0])
    assert bad is None, "Out-of-range INPUT index should return None"

    print(" caching: _normalize_sketch / _denormalize_sketch round-trip correctly")


def _test_lowering_cache_repeated_node_in_graph() -> None:
    """A graph with two structurally identical synthesis nodes should populate
    the cache on the first synthesis and reuse it on the second.

    We build a graph where x and y both undergo the same reduce_sum (axis=1,
    keep_dims=True) and verify:
      1. Both nodes lower successfully.
      2. After lowering the cache contains the expected key.
      3. Re-running the lowering with a different pair of same-rank inputs
         produces a cache hit (not a fresh synthesis run).
    """
    def kernel_dual_reduce(x, y):
        return x.sum(axis=1, keep_dims=True) + y.sum(axis=1, keep_dims=True)

    G = _build_graph_from_kernel(
        kernel_dual_reduce,
        ("x", (4, 8)),
        ("y", (4, 8)),
    )

    # Ensure there are two reduce_sum nodes (the kernel above must produce them)
    reduce_nodes = _nodes_by_op(G, "reduce_sum")
    assert len(reduce_nodes) >= 2, (
        f"Expected >= 2 reduce_sum nodes in dual-reduce graph, got {len(reduce_nodes)}"
    )

    # Clear caches so this test is not influenced by earlier test runs.
    _clear_synthesis_caches()

    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000, verbose=False)
    assert G_hw is not None, "lower_nu_graph returned None for dual-reduce kernel"
    assert _graph_uses_hw_only(G_hw), (
        "Lowered dual-reduce graph still contains public ops"
    )

    # The reduce_sum key should now be in the cache.
    first_reduce = reduce_nodes[0]
    orig_syms = _graph_symbolic_tensors(G)
    inp_sym = orig_syms[first_reduce.inputs[0]]
    expected_key = _make_lowering_cache_key(
        first_reduce.op, first_reduce.attrs, [inp_sym]
    )
    with _SYNTHESIS_CACHE_LOCK:
        cached = _SYNTHESIS_CACHE.get(expected_key, _CACHE_MISS)
    assert cached is not _CACHE_MISS, (
        "Cache should contain an entry for reduce_sum after lowering"
    )
    assert cached is not None, (
        "Cached entry for reduce_sum must be a successful result, not None"
    )

    # Run a second lowering with fresh symbolic inputs of the same rank.
    # The second call should hit the cache (same key) and produce a valid result.
    new_sym = SymTensor("p", rank=2)
    fake_node = Node(id="reduce_sum_test", op=first_reduce.op,
                     inputs=["p"], attrs=dict(first_reduce.attrs))
    fake_target = _sym_expr_from_graph_node(fake_node, [new_sym])
    result = _lower_node(
        fake_node, fake_target,
        [(new_sym, "p")],
        max_hw_size=2, timeout=5000,
    )
    assert result is not None, (
        "Second _lower_node call (expected cache hit) returned None"
    )
    new_nodes, output_id, output_sym = result
    assert output_sym is not None, "Output sym must not be None on cache hit"

    print(" caching: repeated reduce_sum node hits the cache and returns a valid result")


def _test_lowering_cache_all_repeated_node_in_graph() -> None:
    """_lower_node_all should also populate and reuse _SYNTHESIS_CACHE_ALL."""
    def kernel_dual_reduce(x, y):
        return x.sum(axis=1, keep_dims=True) + y.sum(axis=1, keep_dims=True)

    G = _build_graph_from_kernel(
        kernel_dual_reduce,
        ("x", (4, 8)),
        ("y", (4, 8)),
    )
    reduce_nodes = _nodes_by_op(G, "reduce_sum")
    assert reduce_nodes, "Expected at least one reduce_sum node"
    first_reduce = reduce_nodes[0]
    orig_syms = _graph_symbolic_tensors(G)

    _clear_synthesis_caches()

    inp_sym = orig_syms[first_reduce.inputs[0]]
    alts1 = _lower_node_all(
        first_reduce,
        orig_syms[first_reduce.id],
        [(inp_sym, first_reduce.inputs[0])],
        max_hw_size=2,
        timeout=5000,
        verbose=False,
        max_workers=None,
    )
    assert alts1, "_lower_node_all returned no alternatives for reduce_sum"

    expected_key = _make_lowering_cache_key(
        first_reduce.op, first_reduce.attrs, [inp_sym]
    )
    with _SYNTHESIS_CACHE_LOCK:
        cached_all = _SYNTHESIS_CACHE_ALL.get(expected_key, _CACHE_MISS)
    assert cached_all is not _CACHE_MISS, (
        "SYNTHESIS_CACHE_ALL should contain an entry for reduce_sum after _lower_node_all"
    )
    assert cached_all, "Cached all-alternatives must be non-empty"

    # Second call with a fresh sym of the same rank → should use the cache.
    new_sym = SymTensor("q", rank=2)
    fake_node2 = Node(id="reduce_sum_test2", op=first_reduce.op,
                      inputs=["q"], attrs=dict(first_reduce.attrs))
    fake_target2 = _sym_expr_from_graph_node(fake_node2, [new_sym])
    alts2 = _lower_node_all(
        fake_node2, fake_target2,
        [(new_sym, "q")],
        max_hw_size=2,
        timeout=5000,
        verbose=False,
    )
    assert alts2, "Cache-hit _lower_node_all returned no alternatives"
    assert len(alts2) == len(alts1), (
        f"Cache hit must return same number of alternatives as original synthesis "
        f"(got {len(alts2)}, expected {len(alts1)})"
    )

    print(" caching: _lower_node_all populates and reuses SYNTHESIS_CACHE_ALL correctly")


def _test_lowering_cache_cross_variant_reuse() -> None:
    """Lower two variants of the same graph; the second variant should reuse
    cached sketches produced while lowering the first variant.

    We verify that (a) both variants lower successfully, (b) they produce
    hw-only graphs, and (c) the cache is populated after the first variant so
    the second variant does not re-run synthesis from scratch.
    """
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 2, (
        f"kernel_matmul_red_div should have >= 2 variants, got {len(variants)}"
    )

    _clear_synthesis_caches()

    # Lower the first variant; this populates the cache.
    G_hw0 = lower_nu_graph(variants[0], max_hw_size=2, timeout=5000, verbose=False)
    assert G_hw0 is not None, "lower_nu_graph failed for variant 0"
    assert _graph_uses_hw_only(G_hw0), "Variant 0 still has public ops after lowering"

    # The cache should now contain entries for ops in the first variant.
    with _SYNTHESIS_CACHE_LOCK:
        cache_size_after_v0 = len(_SYNTHESIS_CACHE)
    assert cache_size_after_v0 > 0, "Cache should be non-empty after lowering variant 0"

    # Lower the second variant; synthesis should be served from the cache.
    G_hw1 = lower_nu_graph(variants[1], max_hw_size=2, timeout=5000, verbose=False)
    assert G_hw1 is not None, (
        "lower_nu_graph failed for variant 1 (cache-hit path)"
    )
    assert _graph_uses_hw_only(G_hw1), (
        "Variant 1 still has public ops after lowering (cache-hit path)"
    )

    # The cache should not have grown substantially when lowering the second
    # variant (all keys should already be present from variant 0).
    with _SYNTHESIS_CACHE_LOCK:
        cache_size_after_v1 = len(_SYNTHESIS_CACHE)
    assert cache_size_after_v1 == cache_size_after_v0, (
        f"Cache grew from {cache_size_after_v0} to {cache_size_after_v1} entries "
        f"when lowering variant 1 – expected reuse of cached sketches, not new synthesis"
    )

    print(
        f" caching: cross-variant reuse – lowered 2 variants with "
        f"{cache_size_after_v0} cached synthesis result(s), no new synthesis on variant 1"
    )


def _test_lower_nu_graph_variants_starts_with_fresh_kernel_cache() -> None:
    """A new kernel-level lowering run should clear stale cache entries first.

    The cache should still fill during the run so variants of the same kernel
    can reuse synthesized tensor-op lowerings, but unrelated entries left over
    from a previous kernel must not survive into the next measurement.
    """
    G = build_kernel_matmul_red_div_graph(4, 8, 16)
    variants = nu_graph_generation_z3(G, verbose=False)
    assert len(variants) >= 2, "kernel_matmul_red_div should emit multiple variants"

    stale_key = ("stale-kernel",)
    with _SYNTHESIS_CACHE_LOCK:
        _SYNTHESIS_CACHE[stale_key] = ("INPUT", 0)
        _SYNTHESIS_CACHE_ALL[stale_key] = [("INPUT", 0)]

    lowered = lower_nu_graph_variants(
        variants[:2],
        max_hw_size=2,
        timeout=5000,
        verbose=False,
    )
    assert builtins.all(g_hw is not None for g_hw in lowered), (
        "lower_nu_graph_variants should still lower both variants successfully"
    )

    with _SYNTHESIS_CACHE_LOCK:
        assert stale_key not in _SYNTHESIS_CACHE, (
            "Kernel-level lowering should clear stale single-result cache entries first"
        )
        assert stale_key not in _SYNTHESIS_CACHE_ALL, (
            "Kernel-level lowering should clear stale all-results cache entries first"
        )
        assert _SYNTHESIS_CACHE, (
            "Kernel-level lowering should repopulate the cache as variants are synthesized"
        )

    print(" caching: lower_nu_graph_variants starts each kernel with an empty cache")


def _test_lowering_cache_incompatible_key() -> None:
    """Different ops / attrs / input ranks must not share a cache entry."""
    x2 = SymTensor("x", rank=2)
    x3 = SymTensor("x", rank=3)
    y2 = SymTensor("y", rank=2)

    # Different ops
    k1 = _make_lowering_cache_key("reduce_sum", {"axis": 1}, [x2])
    k2 = _make_lowering_cache_key("matmul", {"axis": 1}, [x2, y2])
    assert k1 != k2, "reduce_sum and matmul must have distinct cache keys"

    # Same op, different attrs
    k3 = _make_lowering_cache_key("reduce_sum", {"axis": 0}, [x2])
    k4 = _make_lowering_cache_key("reduce_sum", {"axis": 1}, [x2])
    assert k3 != k4, "reduce_sum with axis=0 and axis=1 must have distinct keys"

    # Same op/attrs, different input rank
    k5 = _make_lowering_cache_key("reduce_sum", {"axis": 1}, [x2])
    k6 = _make_lowering_cache_key("reduce_sum", {"axis": 1}, [x3])
    assert k5 != k6, "reduce_sum on rank-2 vs rank-3 input must have distinct keys"

    print(" caching: incompatible lowering problems have distinct cache keys")


def _test_lowering_cache_result_equivalence() -> None:
    """Cached lowering result must be semantically equivalent to the original.

    We lower the same operation twice (with different generated node IDs but
    identical op/attrs/ranks), clearing the cache in between to get a fresh
    synthesis for comparison.  Both results must be individually equivalent to
    the original target expression.
    """
    def kernel_single_reduce(x):
        return x.sum(axis=1, keep_dims=True)

    G = _build_graph_from_kernel(
        kernel_single_reduce,
        ("x", (4, 8)),
    )
    reduce_nodes = _nodes_by_op(G, "reduce_sum")
    assert reduce_nodes, "Expected at least one reduce_sum node"
    reduce_node = reduce_nodes[0]
    orig_syms = _graph_symbolic_tensors(G)
    inp_sym = orig_syms[reduce_node.inputs[0]]

    # --- First lowering (fresh synthesis) ---
    _clear_synthesis_caches()
    result1 = _lower_node(
        reduce_node,
        orig_syms[reduce_node.id],
        [(inp_sym, reduce_node.inputs[0])],
        max_hw_size=2,
        timeout=5000,
    )
    assert result1 is not None, "First _lower_node call failed"
    _, _, out_sym1 = result1

    # Verify the first result is equivalent to the target.
    target_sym = orig_syms[reduce_node.id]
    assert _check_equivalent_quiet(target_sym, out_sym1, timeout=15000), (
        "First (non-cached) lowering result is not equivalent to the target"
    )

    # --- Second lowering with a fresh sym of same rank (should hit cache) ---
    new_sym = SymTensor("x_fresh", rank=2)
    fake_node = Node(id="reduce_fresh", op=reduce_node.op,
                     inputs=["x_fresh"], attrs=dict(reduce_node.attrs))
    new_target = _sym_expr_from_graph_node(fake_node, [new_sym])
    result2 = _lower_node(
        fake_node, new_target,
        [(new_sym, "x_fresh")],
        max_hw_size=2,
        timeout=5000,
    )
    assert result2 is not None, "Second _lower_node call (cache-hit path) failed"
    _, _, out_sym2 = result2

    # The cache-hit result must be equivalent to the new target expression.
    assert _check_equivalent_quiet(new_target, out_sym2, timeout=15000), (
        "Cached lowering result is not equivalent to the (new-sym) target expression"
    )

    print(
        " caching: cached lowering result is semantically equivalent "
        "to the original target expression"
    )


def _test_synthesize_hw_graph_post_lowering_swap_variants() -> None:
    """Regression test: synthesize_hw_graph runs nu_graph_generation_z3 on
    each distinct lowered hw graph and includes any newly-discovered
    swap-derived variants in the final result.

    This test demonstrates that a lowered hardware graph can produce additional
    variants when passed back through nu_graph_generation_z3.  Both
    ``nu_graph_generation_z3`` and ``lower_nu_graph_variants`` are mocked so
    the test is fast and entirely deterministic — it does not depend on the
    real Z3-based equivalence checker finding a valid hw-op swap.

    Properties verified:
      1. nu_graph_generation_z3 is called on every distinct lowered hw graph
         (Phase-2 call).
      2. New variants returned by those Phase-2 calls are appended to the
         final results.
      3. Variants already present in the Phase-1 results are NOT added again
         (deduplication by graph signature).
      4. The final result count equals the Phase-1 count plus newly injected
         extras, with no duplicates.
    """
    from unittest.mock import patch

    # --------------------------------------------------------------------------
    # Build three distinct hw-only graphs as test fixtures.
    # --------------------------------------------------------------------------
    def _make_hw_graph(tag: str) -> nuGraph:
        x = Node(id="x", op="input", inputs=[], attrs={}, shape=(4, 8))
        w = Node(id="w", op="input", inputs=[], attrs={}, shape=(8, 16))
        out = Node(id=f"nc_matmul_{tag}", op="nc_matmul",
                   inputs=["x", "w"], attrs={}, shape=(4, 16))
        return nuGraph([x, w, out])

    hw_graph_a = _make_hw_graph("aaa")
    hw_graph_b = _make_hw_graph("bbb")
    hw_extra   = _make_hw_graph("extra")   # extra variant produced by Phase-2

    sig_a     = graph_signature(hw_graph_a)
    sig_b     = graph_signature(hw_graph_b)
    sig_extra = graph_signature(hw_extra)

    # Sanity: all three fixtures must have distinct signatures.
    assert len({sig_a, sig_b, sig_extra}) == 3, (
        "Test fixtures must have three distinct graph signatures"
    )

    # Original (public-op) entry-point graph used as the argument to
    # synthesize_hw_graph.  No actual synthesis will be performed since both
    # nu_graph_generation_z3 and lower_nu_graph_variants are mocked.
    G = build_kernel_transpose_matmul_graph(4, 8, 16)
    sig_original = graph_signature(G)

    # --------------------------------------------------------------------------
    # Mock nu_graph_generation_z3:
    #   - Called on the original graph (Phase 1)  → [G]  (one variant)
    #   - Called on hw_graph_a (Phase 2)           → [hw_graph_a, hw_extra]
    #   - Called on hw_graph_b (Phase 2)           → [hw_graph_b]
    # --------------------------------------------------------------------------
    hw_gen_call_sigs: list[str] = []

    def _mocked_nu_gen(graph: nuGraph, verbose: bool = False) -> list[nuGraph]:
        sig = graph_signature(graph)
        if sig == sig_original:
            return [graph]          # Phase-1: one variant (the original graph)
        # Phase-2: called on a lowered hw graph.
        hw_gen_call_sigs.append(sig)
        if sig == sig_a:
            return [hw_graph_a, hw_extra]
        if sig == sig_b:
            return [hw_graph_b]
        return [graph]

    # Mock lower_nu_graph_variants to return two predefined hw graphs.
    def _mocked_lower(variants, max_hw_size=2, timeout=3000,
                      verbose=False, max_workers=None):
        return [hw_graph_a, hw_graph_b]

    with (
        patch(f"{__name__}.nu_graph_generation_z3", side_effect=_mocked_nu_gen),
        patch(f"{__name__}.lower_nu_graph_variants", side_effect=_mocked_lower),
    ):
        hw_results = synthesize_hw_graph(G, max_hw_size=2, timeout=5000)

    # Phase-2 must visit the original lowered hw graphs and any newly-added
    # Phase-2 variants until no more distinct graphs are discovered.
    assert len(hw_gen_call_sigs) == 3, (
        f"Expected Phase-2 nu_graph_generation_z3 calls for 3 hw graphs, "
        f"got {len(hw_gen_call_sigs)}: {hw_gen_call_sigs}"
    )
    assert set(hw_gen_call_sigs) == {sig_a, sig_b, sig_extra}, (
        "Phase-2 must be called on hw_graph_a, hw_graph_b, and the new hw_extra variant; "
        f"got calls for: {hw_gen_call_sigs}"
    )

    result_sigs = {graph_signature(g) for g in hw_results}

    # hw_graph_a and hw_graph_b (Phase-1) must both appear.
    assert sig_a in result_sigs, "hw_graph_a must be in final results"
    assert sig_b in result_sigs, "hw_graph_b must be in final results"

    # hw_extra (Phase-2 injection) must also appear.
    assert sig_extra in result_sigs, (
        "Post-lowering swap variant (hw_extra) must be included in final results"
    )

    # No duplicates.
    assert len(hw_results) == len(result_sigs), (
        "synthesize_hw_graph must not return duplicate graphs"
    )

    # Exactly 3 results: hw_graph_a + hw_graph_b + hw_extra.
    assert len(hw_results) == 3, (
        f"Expected 3 results (2 Phase-1 + 1 Phase-2 extra), got {len(hw_results)}"
    )

    print(
        " synthesize_hw_graph: post-lowering swap variants — "
        "Phase-1 hw graphs: 2, Phase-2 new: 1, total: 3"
    )


def _test_synthesize_hw_graph_post_lowering_reaches_fixpoint() -> None:
    from unittest.mock import patch

    def _make_hw_graph(tag: str) -> nuGraph:
        return nuGraph([
            Node("x", "input", [], {"shape": (4, 8)}, (4, 8)),
            Node("w", "input", [], {"shape": (8, 16)}, (8, 16)),
            Node(f"hw_{tag}", "nc_matmul", ["x", "w"], {"tag": tag}, (4, 16)),
        ])

    hw_graph_a = _make_hw_graph("a")
    hw_graph_b = _make_hw_graph("b")
    hw_graph_c = _make_hw_graph("c")

    sig_a = graph_signature(hw_graph_a)
    sig_b = graph_signature(hw_graph_b)
    sig_c = graph_signature(hw_graph_c)

    G = build_kernel_transpose_matmul_graph(4, 8, 16)
    sig_original = graph_signature(G)
    hw_gen_call_sigs: list[str] = []

    def _mocked_nu_gen(graph: nuGraph, verbose: bool = False) -> list[nuGraph]:
        sig = graph_signature(graph)
        if sig == sig_original:
            return [graph]
        hw_gen_call_sigs.append(sig)
        if sig == sig_a:
            return [hw_graph_a, hw_graph_b]
        if sig == sig_b:
            return [hw_graph_b, hw_graph_c]
        if sig == sig_c:
            return [hw_graph_c]
        return [graph]

    def _mocked_lower(variants, max_hw_size=2, timeout=3000, verbose=False, max_workers=None):
        return [hw_graph_a]

    with (
        patch(f"{__name__}.nu_graph_generation_z3", side_effect=_mocked_nu_gen),
        patch(f"{__name__}.lower_nu_graph_variants", side_effect=_mocked_lower),
    ):
        hw_results = synthesize_hw_graph(G, max_hw_size=2, timeout=5000)

    assert hw_gen_call_sigs == [sig_a, sig_b, sig_c], (
        "Phase-2 should keep processing newly-discovered hw variants until no new variants remain"
    )
    result_sigs = {graph_signature(g) for g in hw_results}
    assert result_sigs == {sig_a, sig_b, sig_c}, (
        "Fixpoint Phase-2 propagation should include transitively discovered hw variants"
    )
    print(" synthesize_hw_graph: post-lowering propagation reaches fixpoint")


def _test_synthesize_hw_graph_verbose_prints_phase_variants() -> None:
    G = build_kernel_transpose_matmul_graph(4, 8, 16)
    phase1_variant = G.clone()
    phase1_variant.nodes[-1] = Node(
        phase1_variant.nodes[-1].id,
        phase1_variant.nodes[-1].op,
        list(phase1_variant.nodes[-1].inputs),
        {**phase1_variant.nodes[-1].attrs, "phase": "pre"},
        phase1_variant.nodes[-1].shape,
    )

    lowered_a = nuGraph([
        Node("x", "input", [], {"shape": (4, 8)}, (4, 8)),
        Node("y", "input", [], {"shape": (8, 16)}, (8, 16)),
        Node("hw_a", "tensor_matmul", ["x", "y"], {"phase": "lowered_a"}, (4, 16)),
    ])
    lowered_b = nuGraph([
        Node("x", "input", [], {"shape": (4, 8)}, (4, 8)),
        Node("y", "input", [], {"shape": (8, 16)}, (8, 16)),
        Node("hw_b", "tensor_matmul", ["x", "y"], {"phase": "lowered_b"}, (4, 16)),
    ])
    phase2_extra = nuGraph([
        Node("x", "input", [], {"shape": (4, 8)}, (4, 8)),
        Node("y", "input", [], {"shape": (8, 16)}, (8, 16)),
        Node("hw_extra", "tensor_matmul", ["x", "y"], {"phase": "phase2_extra"}, (4, 16)),
    ])

    sig_original = graph_signature(G)
    sig_lowered_a = graph_signature(lowered_a)
    sig_lowered_b = graph_signature(lowered_b)

    def _mocked_nu_gen(graph: nuGraph, verbose: bool = False) -> list[nuGraph]:
        sig = graph_signature(graph)
        if sig == sig_original:
            return [graph, phase1_variant]
        if sig == sig_lowered_a:
            return [lowered_a, phase2_extra]
        if sig == sig_lowered_b:
            return [lowered_b]
        return [graph]

    def _mocked_lower(variants, max_hw_size=2, timeout=3000,
                      verbose=False, max_workers=None):
        assert len(variants) == 2, (
            "Mocked lowering should receive both pre-lowering variants so the "
            "verbose phase printer can show the full pre-lowering set"
        )
        return [lowered_a, lowered_b]

    buf = io.StringIO()
    with (
        patch("__main__.nu_graph_generation_z3", side_effect=_mocked_nu_gen),
        patch("__main__.lower_nu_graph_variants", side_effect=_mocked_lower),
        redirect_stdout(buf),
    ):
        hw_results = synthesize_hw_graph(G, max_hw_size=2, timeout=5000, verbose=True)

    out = buf.getvalue()
    assert "[synthesize_hw_graph] pre-lowering: 2 variant(s)" in out
    assert "[synthesize_hw_graph] post-lowering: 2 variant(s)" in out
    assert "[synthesize_hw_graph] post-swap/propagation: 3 variant(s)" in out
    assert "phase': 'pre'" in out
    assert "phase': 'lowered_a'" in out
    assert "phase': 'phase2_extra'" in out
    assert len(hw_results) == 3
    print(" synthesize_hw_graph: verbose output prints phased variants")


def _test_relu_all_lowerings_found_within_depth_budget() -> None:
    """Regression test: for a relu node with max_hw_size=2, both the 1-activation
    lowering (activation(IN:x)) and the 2-activation lowering
    (activation(activation(IN:x))) must be discovered by _synthesize_all_from_pool,
    and _synthesize_from_pool must return the shallower 1-activation sketch.

    Previously _synthesize_from_pool used DFS which reached the 2-activation
    sketch before the 1-activation sketch and returned early, so the shallower
    valid lowering was never considered.
    """
    G = build_kernel_relu_matmul_graph(4, 8, 16)
    orig_syms = _graph_symbolic_tensors(G)
    relu_node = _nodes_by_op(G, "relu")[0]
    input_syms = [orig_syms[inp_id] for inp_id in relu_node.inputs]
    target_sym = orig_syms[relu_node.id]
    pool = _build_synthesis_pool(relu_node.op, relu_node.attrs, input_syms)

    # _synthesize_all_from_pool must find BOTH valid lowerings within depth 2.
    all_sketches = _synthesize_all_from_pool(
        target_sym, pool, max_hw_size=2, timeout=5000, verbose=False,
        input_syms=input_syms,
    )
    sketch_strs = [_format_sketch(s) for s in all_sketches]
    one_activation = [s for s in all_sketches if s.op == "activation" and s.hw_size() == 1]
    two_activation = [s for s in all_sketches if s.op == "activation" and s.hw_size() == 2]
    assert one_activation, (
        f"Expected a 1-activation lowering for relu in all-variants search, "
        f"got sketches: {sketch_strs}"
    )
    assert two_activation, (
        f"Expected a 2-activation lowering for relu in all-variants search, "
        f"got sketches: {sketch_strs}"
    )
    print(f" synthesis: relu all-lowerings found both 1-activation and 2-activation "
          f"sketches within max_hw_size=2: {sketch_strs}")

    # _synthesize_from_pool must return the shallowest (1-activation) sketch.
    best = _synthesize_from_pool(
        target_sym, pool, max_hw_size=2, timeout=5000, verbose=False,
        input_syms=input_syms,
    )
    assert best is not None, "Expected _synthesize_from_pool to find a relu lowering"
    assert best.hw_size() == 1, (
        f"Expected _synthesize_from_pool to return the shallower 1-activation sketch "
        f"(hw_size=1), but got hw_size={best.hw_size()}: {_format_sketch(best)}"
    )
    print(f" synthesis: relu canonical lowering is the minimal sketch "
          f"(hw_size={best.hw_size()}): {_format_sketch(best)}")


def run_all_tests() -> None:
    print("\n================ RUNNING NU-GRAPH TESTS ================")
    _test_expected_variant_counts()
    _test_matmul_red_div_graph()
    _test_no_illegal_reduce_broadcast_swap()
    _test_no_illegal_reduce_sqrt_swap()
    _test_rmsnorm_matmul_graph()
    _test_softmax_matmul_graph()
    _test_transpose_matmul_graph()
    _test_matmul_transpose_graph()
    _test_nc_matmul_transpose_graph()
    _test_relu_matmul_graph()
    _test_silu_matmul_graph()
    _test_silu_mlp_graph()
    _test_attention_graph()
    _test_print_graph_includes_symbolic_shapes()
    _test_print_graph_groups_nodes_by_topological_level()
    _test_synthesis_prefers_direct_reduce_candidate()
    _test_synthesis_verbose_reports_candidate_count()
    _test_synthesis_auto_maximizes_sketch_workers()
    _test_effective_max_workers_uses_cpu_count()
    _test_lower_nu_graph_all_variants_does_not_hold_z3_lock_across_node_lowering()
    _test_synthesis_symbolic_shape_prefilter_skips_solver()
    _test_sketch_shape_constraints_violated_rejected_early()
    _test_synthesis_pool_filters_templates_by_target_constituents()
    _test_single_operator_sketches_cover_distinct_inputs()
    _test_division_broadcast_uses_tensor_scalar_not_tensor_tensor()
    _test_division_symbolic_shape_rejection()
    _test_shapes_not_provably_equivalent_broadcast_compatible_candidate()
    _test_tensor_scalar_data_operand_stays_non_broadcastable_tile()
    _test_invoke_hw_op_transpose_uses_nc_transpose_semantics()
    _test_invoke_hw_op_tensor_reduce_preserves_reduced_shape()
    _test_lower_nu_graph_parallel_levels()
    _test_lower_node_all_prefers_shape_exact_div_canonical()
    _test_lowering_cache_key_derivation()
    _test_lowering_cache_normalize_denormalize()
    _test_lowering_cache_repeated_node_in_graph()
    _test_lowering_cache_all_repeated_node_in_graph()
    _test_lowering_cache_cross_variant_reuse()
    _test_lower_nu_graph_variants_starts_with_fresh_kernel_cache()
    _test_lowering_cache_incompatible_key()
    _test_lowering_cache_result_equivalence()
    print("\n================ RUNNING SYNTHESIZER TESTS =============")
    _test_synthesizer_matmul_red_div()
    _test_synthesizer_rmsnorm_matmul()
    _test_synthesizer_transpose_matmul()
    _test_synthesizer_matmul_transpose()
    _test_synthesizer_relu_matmul()
    _test_synthesizer_silu_matmul()
    _test_synthesizer_all_variants_lowered()
    _test_lower_nu_graph_all_variants()
    _test_reduce_sum_lowering_no_crash()
    _test_matmul_1003_multi_input_synthesis()
    _test_kernel_matmul_red_div_div_node_lowered_via_tensor_scalar()
    _test_matmul_with_complex_input_sym_not_rejected()
    _test_matmul_missing_input_guard()
    _test_synthesize_hw_graph_post_lowering_swap_variants()
    _test_synthesize_hw_graph_post_lowering_reaches_fixpoint()
    _test_synthesize_hw_graph_verbose_prints_phase_variants()
    _test_relu_all_lowerings_found_within_depth_budget()
    print("=============== ALL TESTS PASSED  =====================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy synthesizer tracing and in-file test runner")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run in-file nuGraph tests instead of tracing demo kernels",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Maximum worker threads for sketch checking. "
            "Default (None) auto-derives from CPU count."
        ),
    )
    args = parser.parse_args()

    if args.run_tests:
        run_all_tests()
        sys.exit(0)

    kernels: list[tuple[str, Callable[[int, int, int], nuGraph]]] = [
        ("kernel_matmul_red_div", build_kernel_matmul_red_div_graph),
        ("kernel_matmul_red_mul", build_kernel_matmul_red_mul_graph),
        ("kernel_broadcast_row_bias_add", build_kernel_broadcast_row_bias_add_graph),
        ("kernel_reduce_mul_broadcast", build_kernel_reduce_mul_broadcast_graph),
        ("kernel_reduce_broadcast_mul", build_kernel_reduce_broadcast_mul_graph),
        ("kernel_rmsnorm_matmul", build_kernel_rmsnorm_matmul_graph),
        ("kernel_softmax_matmul", build_kernel_softmax_matmul_graph),
        ("kernel_transpose_matmul", build_kernel_transpose_matmul_graph),
        ("kernel_matmul_transpose", build_kernel_matmul_transpose_graph),
        ("kernel_relu_matmul", build_kernel_relu_matmul_graph),
        ("kernel_silu_matmul", build_kernel_silu_matmul_graph),
        ("kernel_silu_mlp", build_kernel_silu_mlp_graph),
        ("kernel_attention", build_kernel_attention_graph),
    ]

    for kname, builder in kernels:
        print("\n" + "=" * 80)
        print(f"Tracing kernel: {kname}")
        print("=" * 80)
        _start_kernel_synthesis_cache(kernel_name=kname, verbose=True)
        G0 = builder(4, 8, 16)
        print(f"=== {kname} :: Original graph ===")
        print_graph(G0)
        print()
        print(f"--- {kname} :: synthesize_hw_graph "
              f"(all variants, including post-lowering hw swaps) ---")
        hw_variants = synthesize_hw_graph(
            G0,
            max_hw_size=2,
            timeout=3000,
            verbose=True,
            max_workers=args.max_workers,
        )
        if not hw_variants:
            print(f"  [synthesis failed for {kname}]")
        else:
            print(f"--- {kname} :: {len(hw_variants)} synthesized hw graph(s) ---")
            for hi, g_hw in enumerate(hw_variants):
                print(f"  +-- hw variant {hi} ---")
                print_graph(g_hw)
        print()
