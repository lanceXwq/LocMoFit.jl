"""LocMoFit.jl's `run_locmofit` for Python (and MATLAB, through `pyenv`), called through juliacall, with the arguments and the return value of LocMoFitPy's `run_locmofit`, so that a caller can switch between the two by changing the import.

juliacall needs a Julia environment that has LocMoFit and PythonCall at juliacall's own version. LocMoFit.jl's environment has both, once the two versions agree (update the older one: `Pkg.update("PythonCall")` there, or juliacall). Point juliacall at it before importing this module:

    PYTHON_JULIACALL_EXE=julia PYTHON_JULIACALL_PROJECT=<LocMoFit.jl folder> python ...

Importing the module starts Julia and loads LocMoFit (a few seconds); the first fit of a session compiles (40–50 s; NOTES.md, Using LocMoFit.jl from SMAPpy).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from juliacall import AnyValue, ArrayValue
from juliacall import Main as jl

jl.seval("using LocMoFit")

_namedtuple = jl.seval(
    "(names, values) -> NamedTuple{Tuple(Symbol.(names))}(Tuple(values))"
)
_matrix = jl.seval("x -> Matrix{Float64}(x)")
_numeric = jl.seval("x -> x isa AbstractArray{<:Real}")
_DTYPES = {np.dtype("float64"): jl.Float64, np.dtype("float32"): jl.Float32}


def run_locmofit(
    model: str,
    locs,
    loc_precisions,
    *,
    init_params: Mapping[str, float] | None = None,
    freeze: str | Sequence[str] = (),
    tie: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    spacing: float = 3.0,
    point_params: Mapping[str, float] | None = None,
    dtype: Any = "float64",
    background: bool = True,
    fov_volume: float | None = None,
    signal_prior: tuple[float, float] | None = (5, 2),
    share_prior: float | None = 5.0,
    history: bool = False,
) -> dict[str, Any]:
    """Fit `model` to one site's localizations with LocMoFit.jl. The arguments and the returned dict are those of LocMoFitPy's `run_locmofit` (its docstring describes them): `locs` and `loc_precisions` are (N, D), and `model_points` is (M, 3). Only `model` differs: the name of a LocMoFit.jl model, spelled exactly (`"Ring"`, `"SphericalCap"`, `"DoubleRing"`, `"SpCapRing"`); model objects, other cases and aliases are LocMoFitPy's only."""
    dtype = np.dtype(dtype)
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be float64 or float32, not {dtype}")
    locs = np.asarray(locs, dtype=np.float64)
    loc_precisions = np.asarray(loc_precisions, dtype=np.float64)
    if loc_precisions.shape != locs.shape:
        raise ValueError(
            f"loc_precisions must match locs shape; got {loc_precisions.shape} vs {locs.shape}"
        )
    if isinstance(tie, Mapping):
        tie = tie.items()
    fit = jl.run_locmofit(
        model,
        _matrix(locs.T),  # LocMoFit.jl takes D×N
        _matrix(loc_precisions.T),
        init_params=_named(init_params),
        freeze=(freeze,) if isinstance(freeze, str) else tuple(freeze),
        tie=tuple((str(t), str(s)) for t, s in tie or ()),
        max_iter=int(max_iter),
        tol=float(tol),
        spacing=float(spacing),
        point_params=_named(point_params),
        dtype=_DTYPES[dtype],
        background=bool(background),
        fov_volume=None if fov_volume is None else float(fov_volume),
        signal_prior=(
            None
            if signal_prior is None
            else tuple(float(v) for v in signal_prior)
        ),
        share_prior=None if share_prior is None else float(share_prior),
        history=bool(history),
    )
    hist = None
    if history:
        thetas = _python(fit.history.parameters)
        hist = {
            "losses": np.array(fit.history.losses, dtype=float),
            "parameters": {
                k: np.array([t[k] for t in thetas]) for k in thetas[0]
            },
        }
    # LocMoFit.jl's weights are a tuple and its model points 3×M
    return {
        "parameters": _python(fit.parameters),
        "signal_fraction": fit.signal_fraction,
        "weights": np.array(fit.weights, dtype=float),
        "components": _python(fit.components),
        "loss": float(fit.loss),
        "model_points": _python(fit.model_points).T,
        "iterations": int(fit.result.stats.iterations),
        "retcode": str(jl.string(fit.result.retcode)),
        "history": hist,
    }


def _named(values: Mapping[str, float] | None):
    """Names and numbers as a Julia NamedTuple."""
    values = dict(values or {})
    return _namedtuple(tuple(values), tuple(float(v) for v in values.values()))


def _python(x):
    """A value of LocMoFit.jl's result as LocMoFitPy gives it: a NamedTuple as a dict, a tuple or vector of them as a list, an array as a NumPy array of floats. Numbers, strings, tuples and `nothing` arrive converted by juliacall."""
    if isinstance(x, tuple):
        return [_python(v) for v in x]
    if isinstance(x, ArrayValue):
        if _numeric(x):
            return np.array(x, dtype=float)
        return [_python(v) for v in x]
    if isinstance(x, AnyValue) and jl.isa(x, jl.NamedTuple):
        return {str(k): _python(v) for k, v in zip(jl.keys(x), x)}
    return x
