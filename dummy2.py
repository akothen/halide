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
# Maps (op, frozen_attrs, input_shape_keys) → None (failure) | _NormSketch (success)
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
    def __init__(
        self,
        node_id: str,
        shape: tuple[int, ...],
        nodes: Optional[list[Any]] = None,
        sym_shape: Optional[tuple[str, ...]] = None,
    ):
        self.node_id = node_id
        self.shape = shape
        self.nodes = list(nodes) if nodes is not None else [_make_input_node(node_id, shape, sym_shape=sym_shape)]

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


def _make_input_node(
    node_id: str,
    shape: tuple[int, ...],
    sym_shape: Optional[tuple[str, ...]] = None,
) -> Node:
    attrs: dict[str, Any] = {"shape": shape}
    if sym_shape is not None:
        attrs["sym_shape"] = tuple(sym_shape)
    return Node(id=node_id, op="input", inputs=[], attrs=attrs)


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


def graph_structure_signature(G: nuGraph) -> str:
    canonical_ids: dict[str, str] = {}
    parts: list[str] = []
    for idx, node in enumerate(G.nodes):
        canonical_id = f"n{idx}"
        canonical_ids[node.id] = canonical_id
        canonical_inputs = ",".join(canonical_ids.get(input_id, input_id) for input_id in node.inputs)
        parts.append(f"{canonical_id}:{node.op}({canonical_inputs})")
    return " | ".join(parts)


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
        sym_shape = node.attrs.get("sym_shape")
        if sym_shape is not None:
            shape = tuple(z3.Int(str(dim)) for dim in sym_shape)
        else:
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
        sig = graph_structure_signature(G_new)
        if sig in seen_graphs:
            return
        seen_graphs.add(sig)
        out.append((G_new, new_pos))

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
    op1: Node,
    op2: Node,
    G_cur: nuGraph,
    G_new: Optional[nuGraph] = None,
    verbose: bool = False,
    timeout: int = 10000,
) -> bool:
    if G_new is None:
        return False
    cur_tensors = _graph_symbolic_tensors(G_cur)
    new_tensors = _graph_symbolic_tensors(G_new)
    lhs = cur_tensors.get(op2.id)
    rhs = new_tensors.get(op2.id)
    if lhs is None or rhs is None:
        return False
    if _graph_uses_hw_only(G_cur) and _graph_uses_hw_only(G_new):
        rhs_node = _node_by_id(G_new, op2.id)
        if rhs_node is None:
            return False
        input_syms = [new_tensors[input_id] for input_id in rhs_node.inputs if input_id in new_tensors]
        if _shape_rejection_reason(lhs, rhs, input_syms=input_syms) is not None:
            return False
    return check_equivalent(
        lhs,
        rhs,
        timeout=timeout,
        rule_name=f"swap_{op1.id}_{op2.id}",
        verbose=verbose,
    )


def nu_graph_generation_z3(G : nuGraph, verbose=False, timeout: int = 10000) -> List[nuGraph]:
    G0 = G.clone()
    M: dict[str, nuGraph] = {graph_structure_signature(G0): G0}
    equivalence_cache: dict[tuple[str, str, str, str], bool] = {}

    for op1_orig in [n for n in G0.nodes if n.op != "input"]:
        M_next: dict[str, nuGraph] = {}
        for G_seed in sorted(M.values(), key=graph_signature):
            pos0 = _position_by_id(G_seed, op1_orig.id)
            if pos0 is None:
                M_next.setdefault(graph_structure_signature(G_seed), G_seed)
                continue

            worklist: list[tuple[nuGraph, int]] = [(G_seed, pos0)]
            visited: set[tuple[str, int]] = set()

            while worklist:
                G_cur, pos = worklist.pop()
                cur_sig = graph_structure_signature(G_cur)
                state_key = (cur_sig, pos)
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
                                cache_key = (cur_sig, str(pos), str(succ_pos), graph_structure_signature(G_new))
                                equivalent = equivalence_cache.get(cache_key)
                                if equivalent is None:
                                    equivalent = z3_equivalent_order(
                                        op1, op2, G_cur, G_new,
                                        verbose=verbose,
                                        timeout=timeout,
                                    )
                                    equivalence_cache[cache_key] = equivalent
                                if not equivalent:
                                    continue
                                M_next.setdefault(graph_structure_signature(G_new), G_new)
                                worklist.append((G_new, p_new))
                                found_valid_swap = True
                                accepted = True
                                break
                            if accepted:
                                break
                        if accepted:
                            break
                if not found_valid_swap:
                    M_next.setdefault(cur_sig, G_cur)
        M.update(M_next)

    return sorted(M.values(), key=graph_signature)


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
        ordered_pairs = list(_iproduct(concrete, concrete))
        # Prefer distinct input pairs before same-input pairs so two-input
        # candidates are explored before degenerate x/x or y/y forms.
        ordered_pairs.sort(key=lambda pair: pair[0] == pair[1])
        for n1, n2 in ordered_pairs:
            templates.append(SketchNode.make_op(
                hw_op,
                [SketchNode.make_op("transpose", [n1], {}), n2],
                attrs,
            ))
            templates.append(SketchNode.make_op(
                hw_op,
                [n1, SketchNode.make_op("transpose", [n2], {})],
                attrs,
            ))
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
    if sketch.op == "activation":
        clean_attrs.setdefault("scale", 1.0)
        clean_attrs.setdefault("bias_const", None)
        clean_attrs.setdefault("with_reduce", False)
    if sketch.op == "activation_reduce":
        clean_attrs.setdefault("scale", 1.0)
        clean_attrs.setdefault("bias_const", None)
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

    The key is built from the target op name, sorted attribute repr pairs, and a
    per-input shape signature that distinguishes concrete dimensions from
    symbolic ones while remaining insensitive to concrete node IDs and z3
    variable names.  This keeps cache reuse across structurally identical graph
    variants, while avoiding reuse of sketches that are only valid for specific
    concrete input shapes.
    """
    def _dim_key(dim: Any) -> tuple[str, Optional[int]]:
        dim_expr = _to_dim(dim)
        if z3.is_int_value(dim_expr):
            return ("const", dim_expr.as_long())
        return ("sym", None)

    frozen_attrs = tuple(sorted((k, repr(v)) for k, v in attrs.items()))
    input_shape_keys = tuple(tuple(_dim_key(dim) for dim in s.shape) for s in input_syms)
    return (op, frozen_attrs, input_shape_keys)


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
        sig = graph_structure_signature(G_hw)
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
    pre-lowering, post-lowering, post-lowering simplification,
    post-swap/propagation, and post-simplification.  Two distinct
    simplification phases exist within the pipeline: post-lowering
    simplification runs immediately after lowering, while post-simplification
    runs after swap/propagation.
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

    Phase 2: immediately simplify the distinct lowered hardware graphs in place
    by collapsing semantically-equivalent nodes/subgraphs when rewiring is safe
    (for 2-node rewrites, the producer must have the consumer as its sole user).

    Phase 3: run ``nu_graph_generation_z3`` again on the lowered + simplified
    hardware graphs to discover additional swap-derived variants among the
    hardware ops.  Any new variants found are appended to the results.
    Variants that increase the node count are included — for example, pushing
    ``nc_transpose`` before ``tensor_scalar`` may require transposing multiple
    inputs and thus structurally grows the graph, but the rewrite is still
    valid.

    Phase 4: simplify the current variant set again after post-lowering
    swap/propagation so newly exposed 2-node compositions are also collapsed.

    All returned graphs are deduplicated by structural graph signature to
    guarantee fixpoint termination.  Pass *verbose=True* to print all sketches
    evaluated during synthesis for manual inspection.
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
        sig = graph_structure_signature(g_hw)
        if sig in seen:
            continue
        seen.add(sig)
        results.append(g_hw)
    if verbose:
        _print_synthesis_phase_variants("post-lowering", results)

    # ------------------------------------------------------------------
    # Phase 2: simplify immediately after lowering
    # ------------------------------------------------------------------
    results = simplify_hw_graph_variants(
        results,
        timeout=timeout,
        verbose=verbose,
    )
    seen.update(graph_structure_signature(g) for g in results)
    if verbose:
        _print_synthesis_phase_variants("post-lowering simplification", results)

    # ------------------------------------------------------------------
    # Phase 3: post-lowering node-swap variant generation on hw graphs
    # ------------------------------------------------------------------
    # Continue exploring newly-discovered hw variants to a fixpoint.  The
    # ``seen`` set (keyed by structural signature) guarantees termination: each
    # distinct graph structure is processed at most once regardless of whether
    # it is larger or smaller than its source graph.  This allows valid rewrites
    # such as moving ``nc_transpose`` past ``tensor_scalar`` even when doing so
    # adds nodes by transposing multiple inputs.
    phase2_variants: list[nuGraph] = []
    phase2_worklist: list[nuGraph] = list(results)
    bounded_post_lowering_swap_timeout = builtins.min(timeout, _POST_LOWERING_SWAP_TIMEOUT_MILLISECONDS)
    for g_hw in phase2_worklist:
        for g_hw_variant in nu_graph_generation_z3(g_hw, timeout=bounded_post_lowering_swap_timeout):
            sig = graph_structure_signature(g_hw_variant)
            if sig not in seen:
                seen.add(sig)
                phase2_variants.append(g_hw_variant)
                phase2_worklist.append(g_hw_variant)
    results.extend(phase2_variants)
    if verbose:
        _print_synthesis_phase_variants("post-swap/propagation", results)

    # ------------------------------------------------------------------
    # Phase 4: simplify again after swap/propagation
    # ------------------------------------------------------------------
    results = simplify_hw_graph_variants(
        results,
        timeout=timeout,
        verbose=verbose,
    )
    if verbose:
        _print_synthesis_phase_variants("post-simplification", results)

    return results


# ---------------------------------------------------------------------------
# Post-simplification helpers
# ---------------------------------------------------------------------------

# Z3 solver timeout values are expressed in milliseconds.  These caps apply
# only to best-effort optimization phases: missing a simplification or
# post-lowering hw swap is preferable to making compilation appear stuck.
_SIMPLIFICATION_EQ_TIMEOUT_MILLISECONDS = 100
_POST_LOWERING_SWAP_TIMEOUT_MILLISECONDS = 50


def _build_general_simplification_pool(
    input_syms: list["SymTensor"],
) -> list["SketchNode"]:
    """Build a synthesis pool over *input_syms* covering all hw op templates.

    Unlike ``_build_synthesis_pool`` this does not filter templates by a target
    op's constituent set, so every single-hw-op sketch is a valid candidate
    regardless of the composition being simplified.
    """
    concrete = [SketchNode.make_input(sym) for sym in input_syms]
    pool: list[SketchNode] = list(concrete)
    for hw_op in _SYNTHESIS_POOL_OP_NAMES:
        # Pass op_name=hw_op so _pool_templates_for_hw_op uses the full
        # constituent set for that op, ensuring all relevant templates are
        # included without filtering by an external target op.
        hw_op_attrs = {"op_name": hw_op}
        templates = _pool_templates_for_hw_op(hw_op, concrete, hw_op_attrs)
        pool.extend(templates)
    return pool


def _two_node_simplification_candidates(G: nuGraph) -> list[tuple[Node, Node]]:
    """Return (producer, consumer) node pairs eligible for simplification.

    A pair qualifies when:
    - Both nodes are graph ops other than ``input``.
    - The producer's *only* user in the graph is the consumer (sole-user
      condition), which guarantees the producer output can be removed safely.
    """
    used_count: dict[str, int] = {}
    for n in G.nodes:
        for inp in n.inputs:
            used_count[inp] = used_count.get(inp, 0) + 1

    node_by_id: dict[str, Node] = {n.id: n for n in G.nodes}
    candidates: list[tuple[Node, Node]] = []

    for consumer in G.nodes:
        if consumer.op == "input":
            continue
        for inp_id in consumer.inputs:
            producer = node_by_id.get(inp_id)
            if producer is None or producer.op == "input":
                continue
            # Sole-user condition: producer's only consumer is this node.
            if used_count.get(producer.id, 0) != 1:
                continue
            candidates.append((producer, consumer))

    return candidates


def _simplify_hw_graph_once(
    G: nuGraph,
    timeout: int = 3000,
    verbose: bool = False,
) -> Optional[nuGraph]:
    """Try a single simplification step on *G*.

    Iterates over all simplification candidates and attempts to replace them with:
    - 0 nodes — if a node/subgraph output is semantically equivalent to one of
      its external inputs (identity/passthrough).
    - 1 node  — if a single hardware op is semantically equivalent to a
      2-node composition.

    Candidate kinds:
    - 1 node  — any non-input node, which may simplify to an external input.
    - 2 nodes — any producer → consumer pair with the sole-user property.

    For 2-node compositions this can replace the composition with either:
    - 0 nodes — if the consumer output is semantically equivalent to one of
      the external inputs (identity/passthrough).
    - 1 node  — if a single hardware op is semantically equivalent to the
      2-node composition.

    Returns a new ``nuGraph`` on the first successful simplification, or
    ``None`` when no simplification is possible.
    """
    candidates = _two_node_simplification_candidates(G)
    bounded_equivalence_check_timeout = builtins.min(timeout, _SIMPLIFICATION_EQ_TIMEOUT_MILLISECONDS)

    try:
        all_syms = _graph_symbolic_tensors(G)
    except KeyError:
        return None

    # ------------------------------------------------------------------
    # Try 1-node replacement (identity / passthrough)
    # ------------------------------------------------------------------
    for node in G.nodes:
        if node.op == "input":
            continue

        external_input_ids: list[str] = []
        seen_ext: set[str] = set()
        for inp_id in node.inputs:
            if inp_id not in seen_ext:
                seen_ext.add(inp_id)
                external_input_ids.append(inp_id)

        if not external_input_ids:
            continue

        external_syms: list[Optional[SymTensor]] = [all_syms.get(eid) for eid in external_input_ids]
        if any(s is None for s in external_syms):
            continue
        ext_syms = [s for s in external_syms if s is not None]

        target_sym = all_syms.get(node.id)
        if target_sym is None:
            continue

        replacement_id: Optional[str] = None
        for ext_id, ext_sym in zip(external_input_ids, ext_syms):
            if _check_equivalent_quiet(target_sym, ext_sym, timeout=bounded_equivalence_check_timeout):
                replacement_id = ext_id
                if verbose:
                    print(
                        f"  [simplify] {node.id}({node.op}): "
                        f"0-node, rewire to {ext_id}"
                    )
                break

        if replacement_id is None:
            continue

        new_graph_nodes: list[Node] = []
        for n in G.nodes:
            if n.id == node.id:
                continue
            new_inputs = [
                replacement_id if inp == node.id else inp
                for inp in n.inputs
            ]
            new_graph_nodes.append(
                Node(n.id, n.op, new_inputs, dict(n.attrs), n.shape)
            )

        new_G = nuGraph(new_graph_nodes)
        try:
            annotate_shapes_concrete(new_G)
        except Exception:
            # Shape annotations are best-effort here; the simplification was
            # proven by symbolic equivalence, so keep the structurally rewired
            # graph even if concrete annotation cannot be recomputed.
            pass
        return new_G

    for producer, consumer in candidates:
        # ------------------------------------------------------------------
        # Identify external inputs to the 2-node subgraph
        # ------------------------------------------------------------------
        external_input_ids: list[str] = []
        seen_ext: set[str] = set()
        for inp_id in producer.inputs:
            if inp_id not in seen_ext:
                seen_ext.add(inp_id)
                external_input_ids.append(inp_id)
        for inp_id in consumer.inputs:
            if inp_id != producer.id and inp_id not in seen_ext:
                seen_ext.add(inp_id)
                external_input_ids.append(inp_id)

        external_syms: list[Optional[SymTensor]] = [all_syms.get(eid) for eid in external_input_ids]
        if any(s is None for s in external_syms):
            continue
        ext_syms = [s for s in external_syms if s is not None]

        target_sym = all_syms.get(consumer.id)
        if target_sym is None:
            continue

        # ------------------------------------------------------------------
        # Try 0-node replacement (identity / passthrough)
        # ------------------------------------------------------------------
        replacement_id: Optional[str] = None
        new_nodes_for_replacement: list[Node] = []

        for ext_id, ext_sym in zip(external_input_ids, ext_syms):
            if _check_equivalent_quiet(target_sym, ext_sym, timeout=bounded_equivalence_check_timeout):
                replacement_id = ext_id
                if verbose:
                    print(
                        f"  [simplify] {producer.id}({producer.op}) -> "
                        f"{consumer.id}({consumer.op}): "
                        f"0-node, rewire to {ext_id}"
                    )
                break

        # ------------------------------------------------------------------
        # Try 1-node replacement
        # ------------------------------------------------------------------
        if replacement_id is None:
            pool = _build_general_simplification_pool(ext_syms)
            sketch = _synthesize_from_pool(
                target_sym, pool, max_hw_size=1, timeout=bounded_equivalence_check_timeout,
                verbose=False, input_syms=ext_syms,
            )
            if sketch is not None:
                sym_to_id: dict[int, str] = {
                    id(sym): eid for eid, sym in zip(external_input_ids, ext_syms)
                }
                raw_nodes: list[Node] = []
                new_root_id = _sketch_to_graph_nodes(sketch, sym_to_id, raw_nodes)
                if new_root_id is not None:
                    replacement_id = new_root_id
                    new_nodes_for_replacement = raw_nodes
                    if verbose:
                        print(
                            f"  [simplify] {producer.id}({producer.op}) -> "
                            f"{consumer.id}({consumer.op}): "
                            f"1-node, sketch={_format_sketch(sketch)}"
                        )

        if replacement_id is None:
            continue

        # ------------------------------------------------------------------
        # Build the simplified graph
        # ------------------------------------------------------------------
        new_graph_nodes: list[Node] = []
        # 0-node identity/passthrough replacements have no nodes to insert;
        # 1-node replacements populate new_nodes_for_replacement above.
        is_zero_node_replacement = not new_nodes_for_replacement
        inserted_replacement = is_zero_node_replacement  # 0-node: nothing to insert

        for n in G.nodes:
            if n.id == producer.id:
                # Insert replacement node(s) at the producer's position
                if new_nodes_for_replacement:
                    new_graph_nodes.extend(new_nodes_for_replacement)
                    inserted_replacement = True
                continue  # drop producer
            if n.id == consumer.id:
                continue  # drop consumer
            # Rewire any node that consumed the consumer's output
            new_inputs = [
                replacement_id if inp == consumer.id else inp
                for inp in n.inputs
            ]
            new_graph_nodes.append(
                Node(n.id, n.op, new_inputs, dict(n.attrs), n.shape)
            )

        # Append replacement nodes at the end if they were never inserted
        # (can happen if producer was the last node — edge case safety)
        if not inserted_replacement and new_nodes_for_replacement:
            new_graph_nodes.extend(new_nodes_for_replacement)

        new_G = nuGraph(new_graph_nodes)
        try:
            annotate_shapes_concrete(new_G)
        except Exception:
            # Shape annotations are best-effort here; the simplification was
            # proven by symbolic equivalence, so keep the structurally rewired
            # graph even if concrete annotation cannot be recomputed.
            pass
        return new_G

    return None


def simplify_hw_graph_variants(
    graphs: list[nuGraph],
    timeout: int = 3000,
    verbose: bool = False,
) -> list[nuGraph]:
    """Apply single-node and 2-node simplification to every graph in *graphs*.

    Each input graph is simplified to a fixpoint and replaces that input
    variant.  Simplification must not produce extra variants; it only returns
    the current simplified variant set, de-duplicated by structural signature.

    Returns the simplified graph for each input graph, or the original graph
    when no simplification applies.
    """
    simplified_variants: list[nuGraph] = []
    seen: set[str] = set()

    for g in graphs:
        current = g
        while True:
            simplified = _simplify_hw_graph_once(current, timeout=timeout, verbose=verbose)
            if simplified is None:
                break
            current = simplified

        if not _graph_uses_hw_only(current):
            continue
        sig = graph_structure_signature(current)
        if sig in seen:
            continue
        seen.add(sig)
        simplified_variants.append(current)

    return simplified_variants


# ---------------------------------------------------------------------------
# Symbolic tiling
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hardware tile-size constants
# ---------------------------------------------------------------------------
# The partition dimension (axis 0) is always 128 for every NKI instruction.
_TILE_PMAX: int = 128

# Sentinel that marks a free dimension whose size is not fixed by hardware.
# The value reuses the existing `tile_size.psum_fmax` class attribute (which
# is `Ellipsis`, a unique non-int object) purely as a distinguishable sentinel;
# its original NKI meaning (maximum free dim on PSUM) is not used here.
# During symbolic_tiling each operator using this sentinel gets its own
# per-operator z3.Int variable constrained to _TILE_FREE_DIM_CHOICES.
_TILE_PSUM_FMAX: Any = tile_size.psum_fmax

# Valid free-dimension tile sizes for instructions without a hardware limit.
# Hardware supports 512, 1024 and 2048 for these ops.
_TILE_FREE_DIM_CHOICES: tuple[int, ...] = (512, 1024, 2048)

# nc_matmul-specific free-dimension limits (hardware-fixed).
# The moving operand's free dim is capped at 512; the stationary operand's
# free dim is capped at 128 (equal to the partition size).  The output tile
# maps axis-0 to the stationary free dim (128) and axis-1 to the moving free
# dim (512), so the overall output tile has t_0=128, t_1=512.
_TILE_NC_MATMUL_MOVING_FMAX: int = 512
_TILE_NC_MATMUL_STATIONARY_FMAX: int = 128  # equals _TILE_PMAX

# nc_transpose: both partition and free dims are 128.
_TILE_NC_TRANSPOSE_FMAX: int = 128  # equals _TILE_PMAX

# SRAM capacity bound used in the block-size constraint:
#   b_0 * t_0 * b_1 * t_1  ≤  _TILE_SRAM_CAP
# Set to the maximum possible single-tile footprint (max free dim × partition).
_TILE_SRAM_CAP: int = builtins.max(_TILE_FREE_DIM_CHOICES) * _TILE_PMAX  # 2048 * 128

# Maps each hw op to its (t_0, t_1) hardware tile dimensions (partition × free axis).
#
#   t_0 = 128  for all ops  (partition dim is always fixed at 128).
#   t_1:
#     • nc_matmul   → 512  (moving-operand free dim limit)
#     • nc_transpose → 128  (free dim limit equals partition)
#     • all others  → _TILE_PSUM_FMAX sentinel; during symbolic_tiling a
#                     per-operator symbolic variable is created and constrained
#                     to _TILE_FREE_DIM_CHOICES = {512, 1024, 2048}.
#
# Ops not present in this table fall back to (_TILE_PMAX, _TILE_PSUM_FMAX).
_HW_TILE_DIMS: dict[str, tuple[Any, Any]] = {
    "activation":              (_TILE_PMAX, _TILE_PSUM_FMAX),
    "activation_reduce":       (_TILE_PMAX, _TILE_PSUM_FMAX),
    "dma_copy":                (_TILE_PMAX, _TILE_PSUM_FMAX),
    "dma_transpose":           (_TILE_PMAX, _TILE_PSUM_FMAX),
    "exponential":             (_TILE_PMAX, _TILE_PSUM_FMAX),
    "nc_matmul":               (_TILE_PMAX, _TILE_NC_MATMUL_MOVING_FMAX),
    "nc_transpose":            (_TILE_PMAX, _TILE_NC_TRANSPOSE_FMAX),
    "reciprocal":              (_TILE_PMAX, _TILE_PSUM_FMAX),
    "tensor_copy":             (_TILE_PMAX, _TILE_PSUM_FMAX),
    "tensor_partition_reduce": (_TILE_PMAX, _TILE_PSUM_FMAX),
    "tensor_reduce":           (_TILE_PMAX, _TILE_PSUM_FMAX),
    "tensor_scalar":           (_TILE_PMAX, _TILE_PSUM_FMAX),
    "tensor_tensor":           (_TILE_PMAX, _TILE_PSUM_FMAX),
}


@dataclass
class TilingParams:
    """Symbolic tiling parameters for a single hardware operator node.

    Attributes:
        op_id:      Node id of the operator.
        tile_dims:  ``(t_0, t_1)`` – hardware tile sizes.  ``t_0`` is always
                    the concrete integer :data:`_TILE_PMAX` (128).  ``t_1`` is
                    either a concrete integer (for ops with a fixed free-dim
                    limit, e.g. ``nc_matmul``, ``nc_transpose``) or the
                    :data:`_TILE_PSUM_FMAX` sentinel (for ops whose free dim
                    is chosen from :data:`_TILE_FREE_DIM_CHOICES`).
        block_dims: ``(b_0, b_1)`` – symbolic block-dimension variables.
        strip_dims: ``(n_0, n_1)`` – derived strip-dimension variables
                    (shared across non-reduction axes).
    """

    op_id: str
    tile_dims: tuple[Any, Any]
    block_dims: tuple[z3.ArithRef, z3.ArithRef]
    strip_dims: tuple[z3.ArithRef, z3.ArithRef]


@dataclass
class GraphTiling:
    """Symbolic tiling assignment for all hw-op nodes in a lowered nuGraph.

    Attributes:
        op_tilings:       Mapping from ``node.id`` → :class:`TilingParams`.
        shared_strip_dims: Shared strip-dim z3 variables, keyed by axis index.
                           Only non-reduction axes appear here.
        ctx:              A :class:`Context` collecting all well-formedness
                          constraints (block ≥ 1, strip formula, SRAM bound).
    """

    op_tilings: dict[str, TilingParams]
    shared_strip_dims: dict[int, z3.ArithRef]
    ctx: Context


def _reduction_axes(node: Node, G: nuGraph) -> frozenset[int]:
    """Return the set of output-tensor axes that are reduced over by ``node``.

    For ``tensor_reduce`` the reduced axis is taken from the node's ``axis``
    attribute.  ``tensor_partition_reduce`` always reduces axis 0 (the
    partition dimension).  All other hw ops produce outputs with no reduced
    axes.
    """
    op = node.op
    if op == "tensor_reduce":
        axis_attr = node.attrs.get("axis", 1)
        axis = int(axis_attr[0] if isinstance(axis_attr, (list, tuple)) else axis_attr)
        return frozenset({axis})
    if op == "tensor_partition_reduce":
        # Partition reduce collapses axis 0 to size 1.
        return frozenset({0})
    return frozenset()


def symbolic_tiling(G: nuGraph) -> GraphTiling:
    """Assign symbolic tiling parameters to every hw-op node in a lowered nuGraph.

    Strip dimensions along non-reduction axes are unified (shared) across all
    operators to model cross-operator data reuse on SBUF/PSUM.

    Returns a :class:`GraphTiling` containing:

    * per-operator :class:`TilingParams`,
    * shared strip-dim z3 variables (one per non-reduction axis), and
    * a :class:`Context` with all tiling well-formedness constraints.
    """
    with _Z3_LOCK:
        hw_nodes = [n for n in G.nodes if n.op in frozenset(_HW_OP_NAMES)]

        try:
            sym_tensors = _graph_symbolic_tensors(G)
        except (KeyError, z3.Z3Exception):
            sym_tensors = {}

        # Shared strip-dim variables, reused across all non-reduction axis ops.
        shared_strip_dims: dict[int, z3.ArithRef] = {}
        ctx = Context()

        # SRAM capacity bound (concrete): max free dim × partition dim.
        sram_cap = z3.IntVal(_TILE_SRAM_CAP)

        op_tilings: dict[str, TilingParams] = {}

        for node in hw_nodes:
            tile_t0, tile_t1 = _HW_TILE_DIMS.get(
                node.op, (_TILE_PMAX, _TILE_PSUM_FMAX)
            )
            # t_0 (partition axis): always the concrete integer 128.
            t0_z3: z3.ArithRef = z3.IntVal(tile_t0)

            # t_1 (free axis): concrete for nc_matmul/nc_transpose; for all
            # other ops a per-operator symbolic variable constrained to
            # _TILE_FREE_DIM_CHOICES = {512, 1024, 2048}.
            if isinstance(tile_t1, int):
                t1_z3: z3.ArithRef = z3.IntVal(tile_t1)
            else:
                t1_z3 = z3.Int(f"t1_{node.id}")
                ctx.add(z3.Or(*[t1_z3 == z3.IntVal(c) for c in _TILE_FREE_DIM_CHOICES]))

            red_axes = _reduction_axes(node, G)

            # Per-operator symbolic block dimensions.
            b0 = z3.Int(f"b0_{node.id}")
            b1 = z3.Int(f"b1_{node.id}")

            # Strip dimensions: shared for non-reduction axes, per-operator
            # for reduction axes.
            if 0 in red_axes:
                n0: z3.ArithRef = z3.Int(f"n0_{node.id}")
            else:
                if 0 not in shared_strip_dims:
                    shared_strip_dims[0] = z3.Int("strip_0")
                n0 = shared_strip_dims[0]

            if 1 in red_axes:
                n1: z3.ArithRef = z3.Int(f"n1_{node.id}")
            else:
                if 1 not in shared_strip_dims:
                    shared_strip_dims[1] = z3.Int("strip_1")
                n1 = shared_strip_dims[1]

            # Tensor shape for dimension bounds.
            sym = sym_tensors.get(node.id)
            if sym is not None and len(sym.shape) >= 2:
                dim0: z3.ArithRef = sym.shape[0]
                dim1: z3.ArithRef = sym.shape[1]
            elif sym is not None and len(sym.shape) == 1:
                dim0 = sym.shape[0]
                dim1 = z3.IntVal(1)
            else:
                dim0 = z3.Int(f"dim0_{node.id}")
                dim1 = z3.Int(f"dim1_{node.id}")

            bt0 = b0 * t0_z3
            bt1 = b1 * t1_z3

            # n_i == ceil(dim_i / (b_i * t_i))
            # Ceiling division: ceil(a/b) == (a + b - 1) / b for positive a, b.
            # The z3 '/' operator on integer-sorted terms performs integer
            # (floor) division, so (a + b - 1) / b gives ceiling division.
            ctx.add(n0 == (dim0 + bt0 - z3.IntVal(1)) / bt0)
            ctx.add(n1 == (dim1 + bt1 - z3.IntVal(1)) / bt1)

            # Well-formedness: block and strip dimensions must be positive.
            ctx.add(b0 >= z3.IntVal(1))
            ctx.add(b1 >= z3.IntVal(1))
            ctx.add(n0 >= z3.IntVal(1))
            ctx.add(n1 >= z3.IntVal(1))

            # No over-tiling: b_i * t_i <= dim_i + t_i - 1
            ctx.add(bt0 <= dim0 + t0_z3 - z3.IntVal(1))
            ctx.add(bt1 <= dim1 + t1_z3 - z3.IntVal(1))

            # Optional SRAM capacity constraint.
            ctx.add(bt0 * bt1 <= sram_cap)

            op_tilings[node.id] = TilingParams(
                op_id=node.id,
                tile_dims=(tile_t0, tile_t1),
                block_dims=(b0, b1),
                strip_dims=(n0, n1),
            )

        return GraphTiling(
            op_tilings=op_tilings,
            shared_strip_dims=shared_strip_dims,
            ctx=ctx,
        )


def tile_nu_graph_variants(
    graphs: list[nuGraph],
) -> list[tuple[nuGraph, GraphTiling]]:
    """Apply :func:`symbolic_tiling` to each graph in a list of lowered variants."""
    return [(G, symbolic_tiling(G)) for G in graphs]


def print_graph_tiling(G: nuGraph, tiling: GraphTiling) -> None:
    """Print each node annotated as ``[n0,n1,b0,b1] op [t0,t1]``."""
    for node in G.nodes:
        if node.id in tiling.op_tilings:
            tp = tiling.op_tilings[node.id]
            n0, n1 = tp.strip_dims
            b0, b1 = tp.block_dims
            t0, t1 = tp.tile_dims
            print(
                f"  [{n0},{n1},{b0},{b1}] {node.op} [{t0},{t1}]"
                f"  (id={node.id})"
            )
        else:
            print(f"  {node.op}  (id={node.id}, no tiling)")


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


# ---------------------------------------------------------------------------
# Symbolic tiling tests
# ---------------------------------------------------------------------------

def _test_symbolic_tiling_matmul() -> None:
    """Build matmul graph, lower it, apply symbolic_tiling, assert shared strip dims."""
    G = build_kernel_matmul_graph(4, 8, 16)
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000)
    assert G_hw is not None, "lower_nu_graph returned None for matmul"

    tiling = symbolic_tiling(G_hw)

    # All hw-op nodes with no reduction axes should reference the same
    # Python objects for strip_0 and strip_1.
    non_red_nodes = [
        n for n in G_hw.nodes
        if n.op in frozenset(_HW_OP_NAMES) and not _reduction_axes(n, G_hw)
    ]
    assert non_red_nodes, "Lowered matmul graph must contain non-reduction hw nodes"

    if len(non_red_nodes) > 1:
        tp_ref = tiling.op_tilings[non_red_nodes[0].id]
        for nd in non_red_nodes[1:]:
            tp = tiling.op_tilings[nd.id]
            assert tp.strip_dims[0] is tp_ref.strip_dims[0], (
                f"Non-reduction axis-0 strip dims must be the same object "
                f"across {non_red_nodes[0].id!r} and {nd.id!r}"
            )
            assert tp.strip_dims[1] is tp_ref.strip_dims[1], (
                f"Non-reduction axis-1 strip dims must be the same object "
                f"across {non_red_nodes[0].id!r} and {nd.id!r}"
            )

    solver = z3.Solver()
    solver.add(*tiling.ctx.facts)
    result = solver.check()
    assert result == z3.sat, f"symbolic_tiling matmul: context should be SAT, got {result}"
    print(" symbolic_tiling: matmul — shared non-reduction strip dims verified, context SAT")


def _test_symbolic_tiling_rmsnorm_matmul() -> None:
    """Lower rmsnorm+matmul graph and verify reduction-axis strip dims are NOT shared."""
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000)
    assert G_hw is not None, "lower_nu_graph returned None for rmsnorm_matmul"

    tiling = symbolic_tiling(G_hw)

    reduction_node_ids = {
        n.id for n in G_hw.nodes
        if n.op in frozenset(_HW_OP_NAMES) and _reduction_axes(n, G_hw)
    }
    non_reduction_node_ids = {
        n.id for n in G_hw.nodes
        if n.op in frozenset(_HW_OP_NAMES) and not _reduction_axes(n, G_hw)
    }

    # If both reduction and non-reduction nodes are present, their strip dims
    # along the reduced axis must be distinct Python objects.
    for rn in G_hw.nodes:
        if rn.id not in tiling.op_tilings:
            continue
        red_axes = _reduction_axes(rn, G_hw)
        if not red_axes:
            continue
        tp_r = tiling.op_tilings[rn.id]
        for nn_id in non_reduction_node_ids:
            if nn_id not in tiling.op_tilings:
                continue
            tp_n = tiling.op_tilings[nn_id]
            for axis in red_axes:
                assert tp_r.strip_dims[axis] is not tp_n.strip_dims[axis], (
                    f"Reduction-axis {axis} strip dim of {rn.id!r} must NOT be "
                    f"shared with non-reduction node {nn_id!r}"
                )

    solver = z3.Solver()
    solver.add(*tiling.ctx.facts)
    result = solver.check()
    assert result == z3.sat, (
        f"symbolic_tiling rmsnorm_matmul: context should be SAT, got {result}"
    )
    print(
        " symbolic_tiling: rmsnorm_matmul — reduction-axis strip dims not shared, "
        "context SAT"
    )


def _test_symbolic_tiling_all_kernels() -> None:
    """Smoke-test symbolic_tiling on all build_kernel_*_graph helpers."""
    kernel_builders: list[tuple[str, Any]] = [
        ("matmul_red_div",        build_kernel_matmul_red_div_graph),
        ("matmul_red_mul",        build_kernel_matmul_red_mul_graph),
        ("rmsnorm_matmul",        build_kernel_rmsnorm_matmul_graph),
        ("transpose_matmul",      build_kernel_transpose_matmul_graph),
        ("matmul_transpose",      build_kernel_matmul_transpose_graph),
        ("relu_matmul",           build_kernel_relu_matmul_graph),
        ("silu_matmul",           build_kernel_silu_matmul_graph),
        ("matmul",                build_kernel_matmul_graph),
        ("rmsnorm",               build_kernel_rmsnorm_graph),
        ("relu",                  build_kernel_relu_graph),
        ("silu",                  build_kernel_silu_graph),
    ]
    for kname, builder in kernel_builders:
        G = builder(4, 8, 16)
        G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000)
        if G_hw is None:
            print(f" symbolic_tiling: {kname} — skipped (lower_nu_graph returned None)")
            continue
        tiling = symbolic_tiling(G_hw)
        solver = z3.Solver()
        solver.add(*tiling.ctx.facts)
        result = solver.check()
        assert result == z3.sat, (
            f"symbolic_tiling {kname}: context should be SAT, got {result}"
        )
        print(
            f" symbolic_tiling: {kname} — "
            f"{len(tiling.op_tilings)} op(s) tiled, context SAT"
        )


def _test_symbolic_tiling_tile_dims() -> None:
    """Verify that the hardware tile-size constants are correctly reflected in
    TilingParams for nc_matmul, nc_transpose, and generic ops.

    Concrete checks:
    - Every op has t_0 == 128 (partition dim).
    - nc_matmul has t_1 == 512 (moving-operand free dim).
    - nc_transpose has t_1 == 128 (free dim equals partition).
    - All other ops have t_1 as the _TILE_PSUM_FMAX sentinel (non-concrete).
    - The solver can assign t_1 ∈ {512, 1024, 2048} for each generic op.
    """
    G = build_kernel_rmsnorm_matmul_graph(4, 8, 16)
    G_hw = lower_nu_graph(G, max_hw_size=2, timeout=5000)
    assert G_hw is not None, "lower_nu_graph returned None for rmsnorm_matmul"

    tiling = symbolic_tiling(G_hw)

    for node in G_hw.nodes:
        if node.op not in frozenset(_HW_OP_NAMES):
            continue
        tp = tiling.op_tilings.get(node.id)
        assert tp is not None, f"Node {node.id!r} missing from op_tilings"

        t0, t1 = tp.tile_dims
        assert t0 == _TILE_PMAX, (
            f"{node.id} ({node.op}): expected t_0 == {_TILE_PMAX}, got {t0}"
        )

        if node.op == "nc_matmul":
            assert t1 == _TILE_NC_MATMUL_MOVING_FMAX, (
                f"nc_matmul {node.id}: expected t_1 == {_TILE_NC_MATMUL_MOVING_FMAX}, got {t1}"
            )
        elif node.op == "nc_transpose":
            assert t1 == _TILE_NC_TRANSPOSE_FMAX, (
                f"nc_transpose {node.id}: expected t_1 == {_TILE_NC_TRANSPOSE_FMAX}, got {t1}"
            )
        else:
            # Generic op: t_1 must be the sentinel (not a concrete int).
            assert not isinstance(t1, int), (
                f"{node.id} ({node.op}): expected sentinel t_1, got concrete {t1!r}"
            )

    # The constraint system for generic ops must allow each valid free-dim
    # choice.  Check that the tiling context is satisfiable overall.
    solver = z3.Solver()
    solver.add(*tiling.ctx.facts)
    result = solver.check()
    assert result == z3.sat, (
        f"symbolic_tiling tile_dims: context should be SAT, got {result}"
    )

    # Verify that the solver model assigns t_1 ∈ {512, 1024, 2048} for a
    # generic op (if any exist in the graph).
    model = solver.model()
    generic_nodes = [
        n for n in G_hw.nodes
        if n.op in frozenset(_HW_OP_NAMES)
        and n.op not in {"nc_matmul", "nc_transpose"}
    ]
    for n in generic_nodes:
        t1_var = z3.Int(f"t1_{n.id}")
        val = model.eval(t1_var, model_completion=True)
        val_int = val.as_long() if isinstance(val, z3.IntNumRef) else None
        assert val_int in _TILE_FREE_DIM_CHOICES, (
            f"Generic op {n.id!r} ({n.op}): solver chose t_1={val_int}, "
            f"expected one of {_TILE_FREE_DIM_CHOICES}"
        )

    print(
        " symbolic_tiling: tile_dims — t_0=128 for all ops, "
        "nc_matmul t_1=512, nc_transpose t_1=128, "
        "generic ops t_1∈{512,1024,2048} verified"
    )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_all_tests() -> None:
    """Run all in-file tests for the nuGraph synthesizer and symbolic tiling."""
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
    print("\n================ RUNNING SYMBOLIC TILING TESTS =========")
    _test_symbolic_tiling_matmul()
    _test_symbolic_tiling_rmsnorm_matmul()
    _test_symbolic_tiling_all_kernels()
    _test_symbolic_tiling_tile_dims()
    print("=============== ALL TESTS PASSED  =====================\n")


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


def kernel_matmul(x: DummyArray, w: DummyArray) -> DummyArray:
    return x @ w


def kernel_tensor_mul(x: DummyArray, y: DummyArray) -> DummyArray:
    return x + y


def kernel_rmsnorm(x: DummyArray) -> DummyArray:
    xx = x * x
    rec = xx.sum(axis=1, keep_dims=True)
    rms = rec.sqrt()
    norm = x / rms
    return norm


def kernel_softmax(x: DummyArray) -> DummyArray:
    ex = x.exp()
    den = ex.sum(axis=1, keep_dims=True)
    probs = ex / den
    return probs


def kernel_relu(x: DummyArray) -> DummyArray:
    return x.relu()


def kernel_silu(x: DummyArray) -> DummyArray:
    return x.silu()


def kernel_silu_mlp(x: DummyArray, w1: DummyArray, w2: DummyArray) -> DummyArray:
    h1 = x @ w1
    h2 = x @ w2
    a = h1.silu()
    h3 = a * h2
    return h3


def kernel_silu_mlp_full(x: DummyArray, w1: DummyArray, w2: DummyArray, w3: DummyArray) -> DummyArray:
    h1 = x @ w1
    h2 = x @ w2
    a = h1.silu()
    h3 = a * h2
    return h3 @ w3


def kernel_relu_mlp(x: DummyArray, w1: DummyArray, w2: DummyArray) -> DummyArray:
    h1 = x @ w1
    h2 = x @ w2
    a = h1.relu()
    h3 = a * h2
    return h3


def kernel_relu_mlp_full(x: DummyArray, w1: DummyArray, w2: DummyArray, w3: DummyArray) -> DummyArray:
    h1 = x @ w1
    h2 = x @ w2
    a = h1.relu()
    h3 = a * h2
    return h3 @ w3


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


def _build_graph_from_kernel(
    kernel: Callable[..., DummyArray],
    *inputs: tuple[str, tuple[int, ...]] | tuple[str, tuple[int, ...], tuple[str, ...]],
) -> nuGraph:
    args: list[DummyArray] = []
    for spec in inputs:
        if len(spec) == 2:
            name, shape = spec
            args.append(DummyArray(name, shape))
            continue
        name, shape, sym_shape = spec
        args.append(DummyArray(name, shape, sym_shape=sym_shape))
    out = kernel(*args)
    return _graph_from_axon_array(out)


def build_kernel_matmul_red_div_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_matmul_red_div,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("y", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_matmul_red_mul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_matmul_red_mul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("y", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_broadcast_row_bias_add_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_broadcast_row_bias_add,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("y", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_reduce_mul_broadcast_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_reduce_mul_broadcast,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("y", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_reduce_broadcast_mul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_reduce_broadcast_mul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("y", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_rmsnorm_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_rmsnorm_matmul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("y", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_softmax_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_softmax_matmul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_transpose_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_transpose_matmul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w", (M, N), ("x_d0", "w_d1")),
    )


def build_kernel_matmul_transpose_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_matmul_transpose,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_relu_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_relu_matmul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_silu_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_silu_matmul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_matmul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_matmul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w", (K, N), ("x_d1", "w_d1")),
    )


def build_kernel_tensor_mul_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_tensor_mul,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("y", (M, K), ("x_d0", "x_d1")),
    )


def build_kernel_rmsnorm_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_rmsnorm,
        ("x", (M, K), ("x_d0", "x_d1")),
    )


def build_kernel_softmax_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_softmax,
        ("x", (M, K), ("x_d0", "x_d1")),
    )


def build_kernel_relu_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_relu,
        ("x", (M, K), ("x_d0", "x_d1")),
    )


def build_kernel_silu_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_silu,
        ("x", (M, K), ("x_d0", "x_d1")),
    )


def build_kernel_silu_mlp_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_silu_mlp,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w1", (K, N), ("x_d1", "w1_d1")),
        ("w2", (K, N), ("x_d1", "w1_d1")),
    )


def build_kernel_silu_mlp_full_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_silu_mlp_full,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w1", (K, N), ("x_d1", "w1_d1")),
        ("w2", (K, N), ("x_d1", "w1_d1")),
        ("w3", (N, N), ("w1_d1", "w3_d1")),
    )


def build_kernel_relu_mlp_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_relu_mlp,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w1", (K, N), ("x_d1", "w1_d1")),
        ("w2", (K, N), ("x_d1", "w1_d1")),
    )


def build_kernel_relu_mlp_full_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_relu_mlp_full,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w1", (K, N), ("x_d1", "w1_d1")),
        ("w2", (K, N), ("x_d1", "w1_d1")),
        ("w3", (N, N), ("w1_d1", "w3_d1")),
    )


def build_kernel_attention_graph(M: int, K: int, N: int) -> nuGraph:
    return _build_graph_from_kernel(
        kernel_attention,
        ("x", (M, K), ("x_d0", "x_d1")),
        ("w_q", (K, N), ("x_d1", "wq_d1")),
        ("w_k", (K, N), ("x_d1", "wk_d1")),
        ("w_v", (K, N), ("x_d1", "wv_d1")),
    )


def print_graph(G: nuGraph, tiling: Optional[GraphTiling] = None) -> None:
    """Print a human-readable representation of *G*.

    When *tiling* is provided each node is additionally annotated with its
    symbolic tiling assignment via :func:`print_graph_tiling`.
    """
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
    if tiling is not None:
        print_graph_tiling(G, tiling)


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
        ("kernel_matmul", build_kernel_matmul_graph),
        ("kernel_tensor_mul", build_kernel_tensor_mul_graph),
        ("kernel_rmsnorm", build_kernel_rmsnorm_graph),
        ("kernel_softmax", build_kernel_softmax_graph),
        ("kernel_relu", build_kernel_relu_graph),
        ("kernel_silu", build_kernel_silu_graph),
        ("kernel_silu_mlp", build_kernel_silu_mlp_graph),
        ("kernel_silu_mlp_full", build_kernel_silu_mlp_full_graph),
        ("kernel_relu_mlp", build_kernel_relu_mlp_graph),
        ("kernel_relu_mlp_full", build_kernel_relu_mlp_full_graph),
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
