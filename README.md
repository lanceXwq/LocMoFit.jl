# LocMoFit

[![Build Status](https://github.com/lanceXwq/LocMoFit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lanceXwq/LocMoFit.jl/actions/workflows/CI.yml?query=branch%3Amain)

> [!WARNING]
> LocMoFit.jl is experimental: it is under active development, and its API may change without notice.

## Overview

LocMoFit.jl fits a geometric model to the localizations of one site in single-molecule localization microscopy (SMLM) data, in 2D or 3D. The models are a ring, a spherical cap, and composites of them, such as two rings (`DoubleRing`) or a cap above a ring (`SpCapRing`). Each model is a dense set of points; a localization's likelihood is a Gaussian mixture over those points, with each localization's own precision and a fitted blur, mixed with a uniform background. The fit maximizes the posterior, with a prior on every fitted quantity, by BFGS with gradients from Enzyme. Starting values are estimated from the localizations unless given.

LocMoFit.jl is a Julia reimplementation of LocMoFit, the MATLAB fitter in [SMAP](https://github.com/jries/SMAP); `MATLAB_DIFFERENCES.md` lists how the two differ. [LocMoFitPy](https://github.com/lanceXwq/LocMoFitPy) is its port to Python (JAX) and gives the same results.

## Installation

LocMoFit.jl is not registered. Install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/lanceXwq/LocMoFit.jl")
```

## Example

Fit a ring to the simulated site in `examples/data` (reading the CSV needs CSV.jl), from the repository's root folder:

```julia
using LocMoFit, CSV

d = CSV.File("examples/data/ring.csv")
locs = permutedims([d.x_nm d.y_nm d.z_nm])              # 3×N positions, nm
precs = permutedims([d.x_err_nm d.y_err_nm d.z_err_nm]) # 3×N precisions (standard deviations), nm

fit = run_locmofit("Ring", locs, precs)
fit.components[1]   # the ring's position, axis and radius
fit.signal_fraction # the share of localizations that are not background
```

The docstring of `run_locmofit` describes its options: the models, starting values, priors, held and tied parameters, 2D data and Float32.

## Documentation

- `run_locmofit`'s docstring (`?run_locmofit`) — how to call a fit and what it returns.
- `examples/` — the tutorial data (simulated with DynMoSim) and scripts that fit them and record a video of each fit.
- `DESIGN.md` — why the code is implemented the way it is, with the measurements behind each choice.
- `MATLAB_DIFFERENCES.md` — how LocMoFit.jl and LocMoFitPy differ from MATLAB LocMoFit.

## License

MIT; see `LICENSE`.
