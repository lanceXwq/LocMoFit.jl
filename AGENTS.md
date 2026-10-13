# AGENTS.md

Guidance for AI coding agents working in this repository. Workspace-wide rules (data layout, units, parity with LocMoFitPy, working with the owner) are in `../AGENTS.md`; read that first.

## What this is

**LocMoFit.jl** fits a *static* parametric geometric model (ring, spherical cap, two rings, cap + ring) to one site's 2D or 3D SMLM localizations by maximum likelihood. The model is a dense set of points ("unit model", scaled/rotated/translated by the parameters); each localization's likelihood is a Gaussian mixture over those points. It is a Julia re-implementation of the MATLAB LocMoFit in `../../MATLAB/SMAP/LocMoFit`, and **LocMoFitPy (`../LocMoFitPy`) mirrors it in JAX** — the two must give the same results.

## Commands

```sh
julia --project=. -e 'using Pkg; Pkg.test()'   # test suite, ~3 min cold (Enzyme precompile), currently FAILS — see Testing
julia --project=. -e 'using LocMoFit'            # quick load check
```

On Apple Silicon, precompilation prints `CpuId` errors ("does not provide information on cpuid leaf 0x00000004"). They come from a dependency and do not stop LocMoFit from loading.

The examples need `CSV`, `DataFrames`, `GLMakie` and `Distributions`, which are not in any project here; they assume the owner's global environment and are run from the repo root (relative CSV paths).

CI and `[compat]` target Julia 1.13 only (see Compatibility in `../AGENTS.md`); do not add code paths for older versions.

## Layout

| Path | Contents |
| --- | --- |
| `src/LocMoFit.jl` | Module, exports, `run_locmofit` (the public entry point) |
| `src/models.jl` | `abstract type Model <: Lux.AbstractLuxLayer`, `build_model(name)`, `fullparams`, `apply_init` |
| `src/ring.jl`, `spcap.jl`, `double_ring.jl`, `spcap_ring.jl` | One model each: `Ring`, `SphericalCap`, `DoubleRing`, `SpCapRing` |
| `src/unit models.jl` | `unit_ring`, `unit_sphere` (Fibonacci), `unit_disk` — note the space in the filename |
| `src/transformations.jl` | `rotmat(θ, ϕ)`, `rot_trans!`, `scale!`, `unit_sphere_to_cap` |
| `src/loss.jl` | `pairwise_sqdist`, `get_neg_exponents`, `negative_log_likelihood`, `optloss`, `get_norm_const`, `sq_hinge` |
| `ext/LocMoFitDistributionsExt.jl` | `simulate_localizations` (loaded with `using Distributions`). Predates SimSMLM. |
| `examples/simulation/`, `examples/fit/`, `examples/plotting/` | Simulate → CSV, fit CSV → MP4 of the fitting process. Tutorial data belong under `examples/` and are committed. `examples/fit/spcap_video_ROI4.jl` runs on experiment data and is gitignored; it still reads the CSV from the repo root, but the file is now in `../data/`. |
| `jlbridge.py` | Python wrapper (juliacall) so MATLAB/Python can call `run_locmofit` |
| `scripts/run_from_matlab.jl` | JSON-over-stdin entry point. **Stale**: calls `LocMoFit.fit_model`, which no longer exists. Untracked. |
| `old/` | Superseded code, untracked. Reference only. |
| Root `simulated_data_*.csv`, `ground_truth_*.csv` | Inputs/outputs of the examples. As tutorial data they belong under `examples/`; they are still at the root and gitignored. |
| Root `*.mp4`, `simulation *.jl`, `scratch.jl`, `fit_with_video_npc.jl` | Owner's scratch (mostly gitignored). Do not edit, include, or test them. |

## How a fit works

`run_locmofit(model_name, locs, loc_precisions; init_params, freeze, max_iter, spacing, dtype, seed)`:

1. `build_model(model_name)` → `getfield(LocMoFit, Symbol(model_name))()`, so **the name string is the struct name**.
2. Parameters = `initialparameters(model, dtype)` overridden by `init_params` (unknown keys warn and are ignored).
3. `unit_model(model, ps; spacing)` builds the unit point set **once, from the initial parameters**. Point count = `ceil(circumference/spacing)` or `ceil(area/spacing²)`. It is fixed for the whole fit; the callback errors if it is mutated. Consequence: the loss value depends on the initial `r`/`c` and `spacing`.
4. `freeze` → `Lux.Experimental.freeze`, which wraps the model in a `FrozenLayer`; frozen values live in `st.frozen_params` and `fullparams(ps, st)` merges them back.
5. `Optimization.solve` with `OptimizationOptimJL.LBFGS()` and `AutoEnzyme()`. Returns `(result, losses, states, model, unit_model)`.

The loss (`optloss`) = mean over localizations of −logsumexp over model points of the per-pair Gaussian log-likelihood, + `penalty(model, θ, st)`. The fitted blur `σ` is added to every localization precision **in quadrature** (`1/(2(s² + σ²))`). For 2D data, only the first two rows of the model points are used.

## Adding or changing a model

A model needs all of: `struct M <: Model end`; `initialparameters(::M, dtype)` returning a `NamedTuple` of scalars; `unit_model(::M, ps; spacing)`; `LuxCore.initialstates`, `parameterlength`, `statelength`; the call `(::M)(unit_model, ps, st) -> (3×K points, st)` using `fullparams(ps, st)`; `penalty(::Union{M, Lux.Experimental.FrozenLayer{<:Any,M}}, ps, st)`; an `include` in `src/LocMoFit.jl`; and an export.

- Write `penalty` against the `Union` with `FrozenLayer`, as `Ring` and `SpCapRing` do. `SphericalCap` and `DoubleRing` dispatch only on the bare type, so any fit with a non-empty `freeze` throws a `MethodError` for them (verified 2026-10-12).
- The forward pass must not mutate its `unit_model` argument (copy first). Enzyme differentiates through it, so keep mutation local to fresh arrays.
- Penalties are soft box constraints via `sq_hinge`: curvature ≤ 1 nm⁻¹, angles in [0, π].
- **Mirror every model, default, penalty, or loss change in LocMoFitPy** (see the parity section in `../AGENTS.md`), or note explicitly that you did not.

## Conventions

- Data are `D×N` matrices (columns are localizations), `D ∈ {2,3}`; precisions are per-axis **standard deviations** of the same shape. Units: nm.
- Default `dtype` is `Float32`; keep functions generic in `T` and convert constants with `T(...)`/`one(T)`.
- Parameter names are Unicode (`θ`, `ϕ`, `α`, `σ`); LocMoFitPy uses ASCII (`theta`, `phi`, `alpha`, `sigma`).
- `.gitignore` ignores scratch only at the repo root (`/*.csv`, `/simulation*.jl`, `/plotting*`, `/scratch.jl`, `/TODO.md`), plus `*.mp4` and `*_video_*` (scripts that run on experiment data). Everything under `examples/` other than those is meant to be committed. Experiment data live in `../data/`.
- Ask before adding dependencies. `PythonCall` is a dependency but is not used in `src/`.

## Testing

- The suite currently fails (1 pass, 2 errors), both test-side:
  - the Julia ≥ 1.12 reference expression yields a `1×M×N` array against an `M×N` result (needs a `dropdims`). The `else` branch for older Julia is unnecessary under the compatibility rule;
  - it calls the 3-argument `pairwise_sqdist(x, y, σ⁻²)`, which was renamed to `get_neg_exponents(x, half_inv_Σ, x₀)` (with a different argument order and meaning).
  Fix the tests, not the source, unless told otherwise.
- There are no fit-level tests. To check a change, run a fit on `simulated_data_*.csv` and compare with LocMoFitPy (same CSV, `init_params`, `spacing`, `dtype`).
- Use `StableRNG(seed)` in tests (`Xoshiro` streams are not stable across Julia versions). Do not tune tolerances to make a test pass.
