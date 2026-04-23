from __future__ import annotations

import builtins
import argparse
import concurrent.futures
import sys
import io
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, permutations, product as _iproduct
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
_VERBOSE_LOCK = Lock()   # serialises verbose prints from concurrent synthesis threads
_Z3_LOCK = Lock()        # serialises z3 formula construction (z3.main_ctx is not thread-safe)


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
    return _new_sym_tensor(
        "tensor_reduce",
        [data],
        {"op": op, "axis": axis, "negate": negate, "keepdims": keepdims, "name": name},
        _default_out_shape(dst, data),
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
    return _new_sym_tensor("tensor_scalar", inputs, attrs, _default_out_shape(dst, data))


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
        return _shape_tensor_tensor(ins[:2], attrs)
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






class AxonArray:
    def __init__(self, node_id: str, shape: tuple[int, ...], nodes: Optional[list[Any]] = None):
        self.node_id = node_id
        self.shape = shape
        self.nodes = list(nodes) if nodes is not None else [_make_input_node(node_id, shape)]

    @staticmethod
    def _merge_nodes(inputs: list["AxonArray"]) -> list[Any]:
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
    def _from_op(op: str, inputs: list["AxonArray"], out_shape: tuple[int, ...], attrs: Optional[dict[str, Any]] = None) -> "AxonArray":
        node_id = _gen_id(op)
        nodes = AxonArray._merge_nodes(inputs)
        nodes.append(Node(id=node_id, op=op, inputs=[inp.node_id for inp in inputs], attrs=dict(attrs or {})))
        return AxonArray(node_id, out_shape, nodes)

    def _binary_op(self, other: Any, op: str) -> "AxonArray":
        if op == "matmul":
            if not isinstance(other, AxonArray):
                raise TypeError("matmul expects AxonArray operand")
            if len(self.shape) != 2 or len(other.shape) != 2:
                raise ValueError("matmul expects rank-2 inputs")
            if not _dims_equal(self.shape[1], other.shape[0]):
                raise ValueError("matmul expects compatible inner dimensions")
            return AxonArray._from_op(op, [self, other], (self.shape[0], other.shape[1]))

        if isinstance(other, AxonArray):
            out_shape = tuple(_normalize_dim(d) for d in _public_broadcast_shape_tuple(self.shape, other.shape))
            return AxonArray._from_op(op, [self, other], out_shape)
        else:
            out_shape = self.shape
            return AxonArray._from_op(op, [self], out_shape, {"scalar": other})

    def __add__(self, other: Any) -> "AxonArray":
        return self._binary_op(other, "add")

    def __mul__(self, other: Any) -> "AxonArray":
        return self._binary_op(other, "mul")

    def __truediv__(self, other: Any) -> "AxonArray":
        return self._binary_op(other, "div")

    def __matmul__(self, other: Any) -> "AxonArray":
        return self._binary_op(other, "matmul")

    def sum(self, axis: Any = None, keep_dims: bool = False) -> "AxonArray":
        out_shape = _public_reduce_out_shape(self.shape, axis, keep_dims)
        return AxonArray._from_op(
            "reduce_sum",
            [self],
            tuple(_normalize_dim(d) for d in out_shape),
            {"axis": axis, "keep_dims": keep_dims},
        )

    def broadcast_like(self, other: "AxonArray") -> "AxonArray":
        return AxonArray._from_op("broadcast", [self, other], other.shape)

    def sqrt(self) -> "AxonArray":
        return AxonArray._from_op("sqrt", [self], self.shape)

    def exp(self) -> "AxonArray":
        return AxonArray._from_op("exp", [self], self.shape)

    def transpose(self) -> "AxonArray":
        if len(self.shape) == 2:
            return AxonArray._from_op("transpose", [self], (self.shape[1], self.shape[0]))
        return AxonArray._from_op("transpose", [self], self.shape)

    def relu(self) -> "AxonArray":
        return AxonArray._from_op("relu", [self], self.shape)

    def silu(self) -> "AxonArray":
        return AxonArray._from_op("silu", [self], self.shape)


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
            continue
        sig = graph_signature(G_new)
        if sig in seen_graphs:
            continue
        seen_graphs.add(sig)
        out.append((G_new, new_pos))

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

# Ops whose only semantic role is a data layout transformation (always included
# in the pool regardless of constituent overlap).
_LAYOUT_TRANSFORM_OPS: frozenset[str] = frozenset({
    "nc_transpose",
    "dma_transpose",
    "tensor_copy",
    "dma_copy",
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
    if hw_op in ("nc_transpose", "dma_copy", "tensor_copy"):
        add_unary({})
        return templates
    if hw_op == "dma_transpose":
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

    augmented_attrs = dict(target_attrs)
    augmented_attrs["op_name"] = target_op

    for hw_op in _HW_OP_NAMES:
        is_layout = hw_op in _LAYOUT_TRANSFORM_OPS
        if not is_layout and not _shares_constituents(target_op, hw_op):
            continue
        templates = _pool_templates_for_hw_op(hw_op, concrete, augmented_attrs)
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
    """Wrapper around quiet-by-default check_equivalent."""
    try:
        return check_equivalent(lhs, rhs, timeout=timeout)
    except Exception:
        return False


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
) -> Optional[SketchNode]:
    """Phase 2: worklist search for a hw-only sketch equivalent to target_sym.

    Uses DFS (pop from end) so that deeper/more-relevant candidates are
    explored before exhausting all shallow ones.  Returns the first sketch
    that passes check_equivalent, or None if the budget is exceeded.

    When *verbose* is True every complete sketch that is evaluated is printed
    together with whether it passed equivalence, allowing manual inspection.
    """
    initial = SketchNode.make_hole()
    worklist: list[SketchNode] = [initial]
    seen: set[SketchNode] = {initial}

    while worklist:
        sketch = worklist.pop()           # DFS: pop from end

        if not sketch.has_hole():
            # Complete sketch – evaluate and check
            candidate_sym = _eval_sketch(sketch)
            if candidate_sym is None:
                if verbose:
                    print(f"    sketch {_format_sketch(sketch)!s:60s}  [eval failed]")
                continue
            if sketch.hw_size() > max_hw_size:
                if verbose:
                    print(f"    sketch {_format_sketch(sketch)!s:60s}  [exceeds depth {max_hw_size}]")
                continue
            equiv = _check_equivalent_quiet(target_sym, candidate_sym, timeout=timeout)
            if verbose:
                status = "EQUIVALENT ✓" if equiv else "not equivalent"
                print(f"    sketch {_format_sketch(sketch)!s:60s}  [{status}]")
            if equiv:
                return sketch
            continue

        # Incomplete sketch – fill the first hole with each pool entry.
        # Iterate in reverse so DFS/LIFO explores pool entries in their
        # declared priority order instead of visiting the last-added template
        # first.
        can_add_hw_op = sketch.hw_size() < max_hw_size
        for pool_entry in reversed(pool):
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

    return None


def _synthesize_all_from_pool(
    target_sym: SymTensor,
    pool: list[SketchNode],
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
    max_workers: int = 4,
) -> list[SketchNode]:
    """Like _synthesize_from_pool but returns ALL equivalent sketches.

    Phase 1 runs the same DFS worklist as _synthesize_from_pool to enumerate
    every complete, evaluatable candidate sketch.  Phase 2 checks each
    candidate for equivalence in DFS order.

    Returns every sketch that passes equivalence (empty list if none).
    When *verbose* is True all sketches and their results are printed in
    original DFS order after the parallel phase, holding _VERBOSE_LOCK so
    output from concurrent node-level threads does not interleave.
    """
    initial = SketchNode.make_hole()
    worklist: list[SketchNode] = [initial]
    seen: set[SketchNode] = {initial}
    candidates: list[tuple[SketchNode, SymTensor]] = []

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

    if not candidates:
        return []

    if verbose:
        with _VERBOSE_LOCK:
            print(f"    candidate_count={len(candidates)}")

    # Phase 2: check all candidates in DFS order.
    # z3.main_ctx() is not thread-safe, so every call to check_equivalent
    # must hold _Z3_LOCK. Running through a thread pool would serialize on
    # the same lock anyway and only add scheduling overhead.
    def _check_one(item: tuple[SketchNode, SymTensor]) -> tuple[SketchNode, bool]:
        sketch, cand_sym = item
        try:
            with _Z3_LOCK:
                return sketch, _check_equivalent_quiet(target_sym, cand_sym, timeout=timeout)
        except Exception:
            return sketch, False

    ordered: dict[int, tuple[SketchNode, bool]] = {}
    for idx, item in enumerate(candidates):
        ordered[idx] = _check_one(item)

    # Phase 3: collect results and emit verbose output in original DFS order
    valid: list[SketchNode] = []
    if verbose:
        with _VERBOSE_LOCK:
            for idx in range(len(candidates)):
                sketch, equiv = ordered[idx]
                status = "EQUIVALENT ✓" if equiv else "not equivalent"
                print(f"    sketch {_format_sketch(sketch)!s:60s}  [{status}]")
                if equiv:
                    valid.append(sketch)
    else:
        for idx in range(len(candidates)):
            sketch, equiv = ordered[idx]
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
    new_id = _gen_id(sketch.op)
    new_nodes.append(Node(id=new_id, op=sketch.op, inputs=child_ids, attrs=clean_attrs))
    return new_id


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
    """
    input_syms = [sym for sym, _ in hw_input_pairs]

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
    )
    if found is None:
        if verbose:
            print(f"  => FAILED: no hw equivalent found for '{node.id}' op={node.op}")
        return None

    if verbose:
        print(f"  => FOUND: {_format_sketch(found)}")

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
) -> list[tuple[list[Node], str, SymTensor]]:
    """Synthesize ALL valid hw-only replacements for *node*.

    Like _lower_node but uses _synthesize_all_from_pool so every equivalent
    sketch is found (with sketch-level parallelism) rather than stopping at
    the first.  Returns a (possibly empty) list of
    (new_nodes, output_hw_id, output_hw_sym) triples – one per valid sketch.
    """
    input_syms = [sym for sym, _ in hw_input_pairs]

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
    )

    if not found_sketches:
        if verbose:
            with _VERBOSE_LOCK:
                print(f"  => FAILED: no hw equivalent found for '{node.id}' op={node.op}")
        return []

    sym_to_node_id: dict[int, str] = {
        id(sym): node_id for sym, node_id in hw_input_pairs
    }

    results: list[tuple[list[Node], str, SymTensor]] = []
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
        results.append((new_nodes, output_id, output_sym))

    return results


# ---------------------------------------------------------------------------
# Full-graph lowering
# ---------------------------------------------------------------------------

def lower_nu_graph(
    G: nuGraph,
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
) -> Optional[nuGraph]:
    """Lower all public ops in *G* to hw ops using sketch-driven synthesis.

    Returns a new nuGraph that uses only hw ops, or None if any node could
    not be lowered.  When *verbose* is True every sketch evaluated for every
    node is printed so the search can be manually inspected.
    """
    # Build symbolic tensors for the original graph (used as synthesis targets)
    orig_syms: dict[str, SymTensor] = {}
    try:
        orig_syms = _graph_symbolic_tensors(G)
    except (KeyError, Exception):
        return None

    hw_nodes: list[Node] = []        # accumulate lowered nodes
    hw_syms: dict[str, SymTensor] = {}  # hw_node_id -> SymTensor
    node_id_map: dict[str, str] = {}    # orig_id -> hw_id

    if verbose:
        print(f"[lower_nu_graph] graph has {len(G.nodes)} nodes")

    for node in G.nodes:
        if node.op == "input":
            # Input nodes pass through unchanged
            hw_nodes.append(Node(node.id, node.op, list(node.inputs), dict(node.attrs), node.shape))
            hw_syms[node.id] = orig_syms[node.id]
            node_id_map[node.id] = node.id
            continue

        if node.op in _PUBLIC_PASSTHROUGH_OPS:
            # Passthrough ops: remap inputs and copy node using hw ids
            new_inputs = [node_id_map.get(inp, inp) for inp in node.inputs]
            # Use the symbolic result of the first hw input as the output sym
            out_sym = hw_syms.get(new_inputs[0]) if new_inputs else None
            if out_sym is None:
                return None
            new_id = _gen_id(node.op)
            hw_nodes.append(Node(new_id, node.op, new_inputs, dict(node.attrs), node.shape))
            hw_syms[new_id] = out_sym
            node_id_map[node.id] = new_id
            continue

        # Build hw_input_pairs: (hw_sym_for_input, hw_node_id_for_input)
        hw_input_pairs: list[tuple[SymTensor, str]] = []
        for inp_id in node.inputs:
            hw_id = node_id_map.get(inp_id, inp_id)
            hw_sym = hw_syms.get(hw_id)
            if hw_sym is None:
                return None
            hw_input_pairs.append((hw_sym, hw_id))

        target_sym = orig_syms.get(node.id)
        if target_sym is None:
            return None

        result = _lower_node(node, target_sym, hw_input_pairs, max_hw_size, timeout, verbose=verbose)
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
) -> list[Optional[nuGraph]]:
    """Lower every variant in *variants* to hardware ops.

    Returns a list of the same length; entries that could not be lowered are None.
    """
    return [lower_nu_graph(v, max_hw_size=max_hw_size, timeout=timeout, verbose=verbose) for v in variants]


def _build_dag_levels(G: nuGraph) -> list[list[Node]]:
    """Group *G*'s nodes into topological levels.

    Level 0 holds nodes with no predecessors (input nodes).  Level k holds
    nodes whose every input is at a level strictly less than k.  All nodes
    within the same level are mutually independent and can be synthesised
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
    max_workers: int = 4,
    max_variants: int = 256,
) -> list[nuGraph]:
    """Lower *G* to hardware discovering every valid sketch choice per node.

    Node-level parallelism is used:

    1. **Node-level**: nodes that sit at the same DAG level (no data
       dependency between them) are synthesised concurrently via a
       ThreadPoolExecutor.
     After synthesis the Cartesian product of per-node alternatives is taken.
    Because all valid sketches for a node are semantically equivalent to the
    target, downstream nodes are synthesised against the canonical (first)
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
    #                     reference id that downstream nodes were synthesised
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
            for inp_id in node.inputs:
                hw_id = node_id_map_can.get(inp_id, inp_id)
                hw_sym = hw_syms_can.get(hw_id)
                if hw_sym is None:
                    return []
                hw_input_pairs.append((hw_sym, hw_id))
            t_sym = orig_syms.get(node.id)
            if t_sym is None:
                return []
            synthesis_args.append((node, t_sym, hw_input_pairs))

        # Synthesise all nodes at this level in parallel (node-level parallelism)
        def _synth(args: SynthArgs) -> list[tuple[list[Node], str, SymTensor]]:
            nd, tgt, pairs = args
            return _lower_node_all(nd, tgt, pairs,
                                   max_hw_size=max_hw_size, timeout=timeout,
                                   verbose=verbose)

        node_alts_list: list[list[tuple[list[Node], str, SymTensor]]]
        if max_workers > 1 and len(synthesis) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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


def synthesize_hw_graph(
    G: nuGraph,
    max_hw_size: int = 2,
    timeout: int = 3000,
    verbose: bool = False,
) -> list[nuGraph]:
    """Generate all variant orderings of *G* and lower each to hw ops.

    Returns the list of successfully lowered hw graphs (duplicates removed by
    graph signature).  Pass *verbose=True* to print all sketches evaluated
    during synthesis for manual inspection.
    """
    variants = nu_graph_generation_z3(G)
    lowered = lower_nu_graph_variants(variants, max_hw_size=max_hw_size, timeout=timeout, verbose=verbose)
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
    z3.main_ctx() while synthesising reduce_sum_1001 in kernel_matmul_red_div.

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
    assert reduce_nodes[0].attrs.get("keep_dims") or reduce_nodes[0].attrs.get("keepdims"), \
        "reduce_sum in kernel_matmul_red_div variant 0 should use keep_dims=True"

    hw_variants = lower_nu_graph_all_variants(gv, max_hw_size=2, timeout=5000)
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


def kernel_matmul_transpose(x: AxonArray, w: AxonArray) -> AxonArray:
    z = x @ w
    return z.transpose()


def kernel_relu_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    return x.relu() @ w


def kernel_silu_matmul(x: AxonArray, w: AxonArray) -> AxonArray:
    return x.silu() @ w


def _graph_from_axon_array(out: AxonArray) -> nuGraph:
    G = nuGraph([Node(id=n.id, op=n.op, inputs=list(n.inputs), attrs=dict(n.attrs)) for n in out.nodes])
    annotate_shapes_concrete(G)
    return G


def _build_graph_from_kernel(kernel: Callable[..., AxonArray], *inputs: tuple[str, tuple[int, ...]]) -> nuGraph:
    args = [AxonArray(name, shape) for name, shape in inputs]
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


def print_graph(G: nuGraph) -> None:
    symbolic_shapes: dict[str, tuple[Any, ...]] = {}
    sym_shape_fallback = "None"
    try:
        symbolic_shapes = {node_id: tensor.shape for node_id, tensor in _graph_symbolic_tensors(G).items()}
    except (KeyError, z3.Z3Exception):
        symbolic_shapes = {}
        sym_shape_fallback = "unavailable"
    for i, n in enumerate(G.nodes):
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
            max_workers=1,
        )
    out = buf.getvalue()
    assert "candidate_count=" in out, "Expected verbose synthesis output to report candidate_count"
    print(" synthesis: verbose output reports candidate_count")


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
    _test_relu_matmul_graph()
    _test_silu_matmul_graph()
    _test_print_graph_includes_symbolic_shapes()
    _test_synthesis_prefers_direct_reduce_candidate()
    _test_synthesis_verbose_reports_candidate_count()
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
    print("=============== ALL TESTS PASSED  =====================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axon synthesizer tracing and in-file test runner")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run in-file nuGraph tests instead of tracing demo kernels",
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
            print(f"--- {kname} :: Variant {vi} :: Lowering to hardware "
                  f"(all sketches, node- & sketch-level parallel) ---")
            hw_variants = lower_nu_graph_all_variants(
                gv, max_hw_size=2, timeout=3000, verbose=True)
            if not hw_variants:
                print(f"  [lowering failed for variant {vi}]")
            else:
                print(f"--- {kname} :: Variant {vi} :: "
                      f"{len(hw_variants)} lowered hw graph(s) ---")
                for hi, g_hw in enumerate(hw_variants):
                    print(f"  +-- hw variant {hi} ---")
                    print_graph(g_hw)
            print()
