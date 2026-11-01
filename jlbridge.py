from typing import Any, Mapping, Optional, Tuple

import numpy as np
from juliacall import Main as jl

jl.seval("using LocMoFit")


def normalize_dtype(dtype):
    if dtype in ("float32", "fp32"):
        return jl.Float32
    if dtype in ("float64", "fp64"):
        return jl.Float64
    return dtype


def _dict_to_namedtuple(d: dict):
    # Returns a Julia NamedTuple value, e.g. (; z=53.0, s=60.0)
    if not d:
        return jl.seval("(;)")
    # Stable order (optional but nice)
    items = sorted(d.items(), key=lambda kv: kv[0])

    # Bind values into Julia Main as temporary variables
    tmp_names = []
    for k, v in items:
        vn = f"__init_{k}"
        setattr(jl, vn, v)  # julia Main.__init_k = v
        tmp_names.append((k, vn))

    expr = "(; " + ", ".join(f"{k}={vn}" for k, vn in tmp_names) + ")"
    nt = jl.seval(expr)
    return nt


def _normalize_freeze(freeze):
    """
    Accept:
      - () / [] / None
      - "z"                     (single string)
      - ("x", "z")              (tuple/list of strings)
      - (jl.Symbol("z"), ...)   (already-symbols)
    Return:
      - Julia tuple of Symbols: (:x, :z)
    """
    if freeze is None:
        return jl.seval("()")

    # If MATLAB passes "z", Python sees a str
    if isinstance(freeze, str):
        names = [freeze]
    else:
        names = list(freeze)

    # Convert each entry to Julia Symbol
    syms = []
    for x in names:
        if isinstance(x, str):
            syms.append(jl.Symbol(x))
        else:
            # assume it's already a Julia Symbol (or convertible)
            syms.append(x)

    # a Julia Tuple (not a Python tuple), to satisfy Tuple{Vararg{Symbol}}
    return jl.tuple(*syms)


def _normalize_tie(tie):
    """
    Accept a mapping {"r2": "r1"} or pairs [("r2", "r1")], each target => source.
    Return a Julia tuple of Symbol pairs: (:r2 => :r1,)
    """
    if not tie:
        return jl.seval("()")
    items = tie.items() if isinstance(tie, Mapping) else tie
    return jl.tuple(
        *[
            jl.Pair(jl.Symbol(str(target)), jl.Symbol(str(source)))
            for target, source in items
        ]
    )


def run_locmofit(
    model_name: str,
    locs: np.ndarray,
    loc_precisions: np.ndarray,
    *,
    init_params: Optional[Mapping[str, Any]] = None,
    freeze: Tuple[str, ...] = (),
    tie: Any = None,
    spacing: float = 3.0,  # 3 nm
    max_iter: int = 200,
    tol: float = 1e-4,
    dtype: str = "float64",
    background: bool = True,
    fov_volume: Optional[float] = None,
    signal_prior: Any = (5, 2),
    history: bool = False,
):
    init_params = _dict_to_namedtuple(dict(init_params or {}))
    freeze_syms = _normalize_freeze(freeze)
    fit = jl.LocMoFit.run_locmofit(
        model_name,
        locs,
        loc_precisions,
        max_iter=max_iter,
        tol=tol,
        init_params=init_params,
        spacing=spacing,
        freeze=freeze_syms,
        tie=_normalize_tie(tie),
        dtype=normalize_dtype(dtype),
        background=background,
        fov_volume=fov_volume,
        signal_prior=(
            jl.tuple(*signal_prior) if signal_prior is not None else None
        ),
        history=history,
    )
    # history is Julia's `nothing` unless history=True; then it holds `losses` and `parameters`, one per iteration
    return {
        "parameters": fit.parameters,
        "signal_fraction": fit.signal_fraction,
        "weights": fit.weights,
        "components": fit.components,
        "loss": fit.loss,
        "model_points": fit.model_points,
        "result": fit.result,
        "history": fit.history,
    }
