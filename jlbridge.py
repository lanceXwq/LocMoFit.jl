import numpy as np
from typing import Tuple, Optional, Mapping, Any
from juliacall import Main as jl
from juliacall import convert

jl.seval("using LocMoFit")


def _dict_to_namedtuple(d: dict):
    # Returns a Julia NamedTuple value, e.g. (; z=53.0, c=0.02)
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
      - "c"                     (single string)
      - ("c", "z")              (tuple/list of strings)
      - (jl.Symbol("c"), ...)   (already-symbols)
    Return:
      - Julia tuple of Symbols: (:c, :z)
    """
    if freeze is None:
        return jl.seval("()")

    # If MATLAB passes "c", Python sees a str
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

    # Build a Julia Tuple (not a Python tuple) to satisfy Tuple{Vararg{Symbol}}
    # `jl.tuple` is the cleanest when available; otherwise we can seval.
    try:
        return jl.tuple(*syms)
    except Exception:
        # fallback: construct in Julia
        setattr(jl, "__freeze_syms", syms)
        t = jl.seval("Tuple(__freeze_syms)")
        delattr(jl, "__freeze_syms")
        return t


def run_locmofit(
    model_name: str,
    locs: np.ndarray,
    loc_precisions: np.ndarray,
    *,
    init: Optional[Mapping[str, Any]] = None,
    seed: int = 1,
    spacing: float = 3.0,  # 3 nm
    freeze: Tuple[str, ...] = (),
    max_iter: int = 200,
    dtype=jl.Float32,
):
    init = {} if init is None else dict(init)
    init_nt = _dict_to_namedtuple(init)
    freeze_syms = _normalize_freeze(freeze)
    res, losses, states, model, umodel = jl.LocMoFit.run_locmofit(
        model_name,
        locs,
        loc_precisions,
        max_iter=max_iter,
        init=init_nt,
        seed=seed,
        spacing=spacing,
        freeze=freeze_syms,
        dtype=dtype,
    )
    return {"result": res, "losses": losses, "states": states}
