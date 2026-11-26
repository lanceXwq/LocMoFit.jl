# Differences from MATLAB LocMoFit

LocMoFit.jl and its JAX port LocMoFitPy re-implement the fitting core of the original MATLAB LocMoFit, which is part of SMAP (`SMAP/LocMoFit`). This file lists where they differ from it:

- features the MATLAB version has and these packages do not;
- deliberate changes;
- issues found in the MATLAB code, which the ports must not copy.

LocMoFit.jl and LocMoFitPy are meant to behave identically, so "the ports" means both, unless a row says otherwise. Since 2026-10-28 LocMoFitPy has LocMoFit.jl's features and gives the same fits (`../LocMoFitPy/JULIA_DIFFERENCES.md`); "LocMoFit.jl" in a row below then holds for LocMoFitPy too.

The comparison is against SMAP commit `d750f5ac` (2026-09-01); LocMoFit itself last changed in `d066594d` (2026-08-28). Status as of 2026-11-01:

- **planned**: decided, not yet implemented;
- **open**: not decided;
- **not planned**: left out for now;
- **by design**: the ports differ on purpose.

## 1. Likelihood

| Topic | MATLAB LocMoFit | LocMoFit.jl / LocMoFitPy | Status |
| --- | --- | --- | --- |
| Signal term | Gaussian mixture over the model points. Each localization has its own diagonal covariance. Points carry weights n, and the mixture is normalized by Σn. | The same mixture without point weights. LocMoFit.jl normalizes it (the mean over each component's points, the components weighted by their shares; `derivations/derivations.qmd`). | — |
| Background | Uniform background per layer, p = (1 − b)·f + b·U. The weight b is fitted in [0.001, 0.999], or reparametrized as a density in localizations/µm². U is the inverse of a fixed volume set by the ROI size d: in 3D a cylinder of diameter d and height d, U = 1/(π(d/2)²·d); in 2D a disc, U = 1/(π(d/2)²). | LocMoFit.jl: p = f₀/V + Σₖ fₖ pₖ, the background as fraction f₀ of one softmax over the background and the components, with a Dirichlet prior on the fractions (default Dirichlet(2, 5, …, 5); for one component the Beta(5, 2) density of the signal fraction 1 − f₀, mode 0.8) and V the volume (area in 2D) of the field of view, `fov_volume` (default: the localizations' bounding box). | done |
| Blur (`variation` / `sigma`) | Added to each precision in quadrature, √(s² + v²), starting (fixed) at 0. | LocMoFit.jl: in quadrature, starting at 1 nm, since the likelihood is stationary in σ at 0. | done |
| Precision floor | For all but `'discrete'` models, precisions below the median are raised to the median (advanced setting `minSigma`). | Every localization keeps its own precision. | by design |
| Missing z precision | If any `locprecznm` is 0, every z precision is replaced by 3 × `locprecnm`. | A precision is required for every axis. | by design |
| Several channels (layers) | Each model belongs to a layer, and each layer has its own background. Layer counts are reweighted by max(N)/N_ℓ (`layerNorm`, on by default), which is not a likelihood. | One channel. | open |
| Objective | A sum of log-likelihoods. There is also an unused `'correlation'` objective (a sum of densities) and a `'fast'` mode that drops terms beyond 3.5σ. | Per localization: LocMoFit.jl minus the log posterior (with the prior on π). The optimum is the same as for the sum. | by design |

## 2. Models

### Generic parameters

| Parameter | MATLAB LocMoFit | LocMoFit.jl / LocMoFitPy | Status |
| --- | --- | --- | --- |
| Position | `x`, `y`, `z` (nm) | `x`, `y`, `z` (nm) | — |
| Orientation | `xrot`, `yrot`, `zrot` in degrees, with R = Rx·Ry·Rz. When all three are free, the fit uses an axis-angle vector instead. | LocMoFit.jl: the axis by two tangent-chart coordinates `vx`, `vy` around the starting axis, given as `theta`, `phi` (no pole; `derivations/derivations.qmd`). There is no rotation about the model's own axis, so only rotationally symmetric models can be fully oriented. | axis: done; rotation about the axis: open |
| Scale | `xscale`, `yscale`, `zscale` (fixed at 1 by default) | none | open |
| Model weight | `weight`, normalized among the models of a layer | LocMoFit.jl: one share per component of a composite, softmax(0, `omega2`, …, `omegaK`), free by default and held with `Fixed()`; the background is fraction 0 of the same softmax, and one Dirichlet prior, by default Dirichlet(2, 5, …, 5), covers it and the components and keeps every fraction above 0 (2026-11-17; before, a Dirichlet(5) on the shares alone, since 2026-11-14); LocMoFitPy the same. | done |

### Model types and sampling

| Topic | MATLAB LocMoFit | LocMoFit.jl / LocMoFitPy | Status |
| --- | --- | --- | --- |
| Point models (`'discrete'`, `'discretized'`) | yes | yes | — |
| Continuous models (rendered on a grid, then interpolated at the localizations) | yes | no | not planned |
| Image models (loaded from a file); a localization set as the model (`locsModel`, `locsImgModel`) | yes | no | not planned |
| Point weights n | yes; used by the 2D side-view projection models and the parametric cap | none: each point is taken to represent an equal area (or length) | open |
| Point spacing | 0.75 × the smallest (median-floored) precision. The point set is regenerated at every evaluation, so the count changes during the fit. | `spacing` in nm (default 3), applied once to the initial parameters. The set stays fixed during the fit, which keeps the objective smooth for gradient-based optimizers. | by design; a helper that picks the spacing from the precisions is planned, at low priority |

### Composite models

MATLAB fits any number of models together (`addModel`). Each model sits in a layer, and the models are mixed by normalized weights. The extrinsic parameters of model k are documented as relative to model 1, but see issue 2 in section 7.

LocMoFit.jl fits any number of simple models together as well: `Composite(components...)`, with component 1 placed by its pose in the world and every other component by its pose in component 1's frame, absolute shape parameters, one shared blur, and ties between parameters (`tie=(:r2 => :r1,)`). `DoubleRing` and `SpCapRing` are presets of it. LocMoFit.jl averages each component's points on its own and weights the components by their shares (`omega2`, …). Until 2026-10-28 LocMoFitPy pooled the points of all components into one set (its `NuclearPoreComplex`), so each component was weighted by its size. That assumes equal labeling density on every component. When the components differ in dimension, as the ring and the cap of `SpCapRing` do, the weights even depend on `spacing`: at a ring radius of 50 nm, the ring gets 2.9% of the points at a 3 nm spacing and 1.5% at 1.5 nm.

Status: done in LocMoFit.jl (2026-10-26). Composite models carry one weight per component: the share of the signal localizations it explains, with each component normalized over its own points. The weights are free by default and can be frozen, for example at the size ratio when equal labeling density is justified.

### Model library

MATLAB LocMoFit lists 23 models (`modelList.m`).

| MATLAB model | Equivalent in the ports |
| --- | --- |
| `ring2D`, `ring3D` | `Ring`; a 2D fit uses its first two coordinates. |
| `sphericalCap3D_surfaceArea`, `sphericalCap3Dp_surfaceArea` | `SphericalCap`. LocMoFit.jl uses the same constant-area cap as MATLAB, parametrized by its scale s (area 4πs²) and, since 2026-11-14, its half-angle α, MATLAB's close angle, with κ′ = sin(α/2), so that 1 − cos α = 2κ′² (0 flat, ±π a full sphere, negative inverted; κ′ = tanh η before, which put the full sphere at η → ∞), with its point count from its own area. Until 2026-10-28 LocMoFitPy used the curvature c and the half-angle α: a flat cap then needed c → 0, and its point count, computed from the full sphere as 4π/(c²·spacing²), was four times what a hemisphere needs and grew without bound. |
| `dualRing3D_discrete` (NPC: 8 corners × 2 copies, ring distance, twist) | `DoubleRing`: two continuous rings with a free relative position, size ratio and tilt. There is no 8-fold symmetry. |
| — | `SpCapRing` (a cap above a ring), a preset composite of `Ring` and `SphericalCap`. MATLAB would build it as a composite too. |
| `arc2D`, `arc2D_arcLen`, `bucket2D` | none |
| `cspline3D_midPoint`, `csplineTube3D_midPoint`, `csplineClosedTube3D_midPoint`, `csplineTube3D_xyz` | none |
| `dualEllipse3D_discrete`, `dualEllipse3D_avgR_discrete`, `ellipse2D`, `ellipse3D` | none |
| `gaussianCluster2D` (a single point; its spread is the blur) | none |
| `hemisphere2DPro`, `hemispheroid2D`, `thickRing2D` (2D side-view projections) | none. In the ports, a 3D model fitted to 2D data plays this role, provided it is sampled at equal area. |
| `spheroidCap3Dp_surfaceArea`, `spheroid3Dp_surfaceArea` | none |
| `locsBG3D` (random background points, for simulation) | none |

The MATLAB folder also has 12 unlisted models (`cylinder3D`, line segments, `simplex2D`, `triangle`, `vertexOfSquare`, `onePoint`, …). Another 39 obsolete ones are kept so that old settings files still load.

## 3. Parameters and optimization

| Topic | MATLAB LocMoFit | LocMoFit.jl / LocMoFitPy | Status |
| --- | --- | --- | --- |
| Bounds | Relative `lb`/`ub` around the start value, clipped to absolute `min`/`max`. They are hard bounds, through `fminsearchbnd`'s transform or `fmincon`'s bounds. | No bounds; priors instead (owner's decision, 2026-11-14: proper priors, no hard bounds). LocMoFit.jl and LocMoFitPy (on main since 2026-11-17): a prior on every quantity, given with its starting value as a `ParamToFit` (`derivations/derivations.qmd`, Priors on every parameter); by default a log-normal on sizes (most probable at 50 nm), which keeps them above 0, and flat on positions, axes, κ′ and the blur. Before, soft quadratic penalties kept sizes ≥ 1 nm. | done (2026-11-17) |
| Fixing parameters | `fix` column | LocMoFit.jl and LocMoFitPy: the prior `Fixed()`, a point mass at the starting value | — |
| Starting values from the data | Converter rules: MATLAB expressions such as `median(locs.xnm)` → `m1.lPar.x`, evaluated once before the fit. | LocMoFit.jl: what `params` leaves open (and is neither held nor tied; `init_params` before 2026-11-17) is estimated from the localizations' moments: the position, axis and shape of a ring or a cap, and for a composite of two of them a search over axes with a score per component type (`derivations/derivations.qmd`, `DESIGN.md`); in 2D the position only. LocMoFitPy: the same (`start.py`), and its SMAPpy plugin leaves the starting values to it. | LocMoFit.jl: done (2026-11-12); LocMoFitPy: done (2026-11-13) |
| Ties between parameters during a fit | none (converter rules only set starting values) | LocMoFit.jl and LocMoFitPy: the prior `Tied(:source)` (`Tied("source")` in LocMoFitPy) | done (2026-11-17) |
| Passing parameters between fits | Converter rules that take another fit as the source; `matchAllPar`; one fitter per step | none | open |
| Cascade | Repeated fits with scaled bounds (and an extra σ offset), each starting from the previous result | none | open |
| Optimizer | `fminsearchbnd` (Nelder–Mead, the default), `fmincon` (finite-difference gradients), `particleswarm` (global search) | Gradients from automatic differentiation. LocMoFit.jl: BFGS with a backtracking line search (Optim.jl, Enzyme), stopping when no gradient component exceeds `tol`. LocMoFitPy runs the same algorithm, transcribed in JAX. | done, as in DynMoFit |
| Multiple starts | none (use `particleswarm`, or random starting values through a converter rule) | none | — |

## 4. Results and diagnostics

| Output | MATLAB LocMoFit | LocMoFit.jl / LocMoFitPy | Status |
| --- | --- | --- | --- |
| Fitted parameters | yes | yes | — |
| Log-likelihood | `LLfit`, per localization | the final loss | — |
| AIC, AICc, normalized AICc | yes | no | open |
| Background density and background-free localization count | yes | the signal fraction (`signal_fraction`), from which both follow | — |
| Expected log-likelihood and z-score (from 1,000 simulations of the fitted model); an "overfitted" control (the localizations as the model) | optional | no | open |
| Confidence intervals | optional; see issue 11 | no | open |
| Derived parameters per model (radius, curvature, area, …) | `getDerivedPars` | yes: each component's shape in `components` (a ring's `r`; a cap's `s`, `kappa`, `curvature`, `radius`, `alpha`, `area`) | done |
| Optimization history | optional, for playback in the viewer | LocMoFit.jl: the loss and the parameters per iteration, with `history=true`. | — |
| Model points | for plotting | returned | — |

## 5. Around the fitter (SMAP)

The MATLAB version runs inside SMAP, which provides:
- the `LocMoFitGUI` plugin: model and parameter tables, a preview, and a fit viewer;
- batch fitting over sites (sequential, through "redraw all");
- per-site storage of the results in the `.sml` file, and tab-delimited export across sites;
- registration and averaging (`locsRegister`, template-free particle fusion);
- galleries and cross-site summary plots;
- simulation of sites from a LocMoFit model (`SimulateSites`, with SMAP's photophysics).

The ports are libraries that fit one site per call. Calling them from SMAP (`LocMoFitPy/run_locmofit.m`, `LocMoFit.jl/jlbridge.py`) and from SMAPpy is separate work. In this workspace, DynMoSim simulates the test data.

SMAP's `WeiMoFit` plugin (`plugins/+ROIManager/+Evaluate/WeiMoFit.m`, 2026-02-24) calls LocMoFitPy through `pyenv`. It expects the old folder `../LocMoFitPy2/` and a `matlab_api` module, neither of which exists in the current package.

## 6. What the ports add

- Gradients by automatic differentiation (Enzyme in Julia, JAX in Python). In LocMoFit.jl, the loop over localizations and model points has its gradient in closed form (`derivations/derivations.qmd`) and is vectorized; every pair is kept, without the truncation of MATLAB's `'fast'` mode. A cap fit of 290 localizations takes about 0.3 s.
- A fixed set of model points, which keeps the objective smooth.
- Float32, and GPU execution through JAX.
- Calls from Julia, Python and MATLAB.

## 7. Issues in MATLAB LocMoFit

These were found by reading the code at SMAP `d750f5ac`. None was run in MATLAB; only the rotation round trip of issue 4 was checked numerically, outside MATLAB. File paths are relative to `SMAP/LocMoFit`. The ports must not reproduce any of these.

### Issues that change fit results

1. **The 2D Gaussian normalization is 4× too large.**
   - Where: `helperFcn/gaussDist2.m:47` computes `2*pi^-1./(xsigma.*ysigma)`, i.e. 2/(πσxσy) instead of 1/(2πσxσy).
   - Effect: the background term is normalized correctly, so a 2D fit maximizes N·log(4 − 3b) plus the correct log-likelihood. That pushes the background fraction b toward zero, which lets background localizations pull on the model. The reported log-likelihood and AIC are off by the same term.
2. **Composite models are placed differently in the fit than in plots and simulations.**
   - Fitting (`@functionModel/modelHandler.m:31-43`) applies model 1's transform first and model k's second: p → T_k(T_1(p)).
   - Plotting and simulation (`getModPoint`, then `@LocMoFit/transModPoint.m`) apply model k's transform first and model 1's last.
   - Unless the two transforms commute, the fitted composite is not the one plotted or simulated. The documentation's "relative to M1" matches the plotting order.
3. **In multi-layer fits, point models are not restricted to their own layer.** `@LocMoFit/intensityCal.m:231-232` weights only the rows of the model's own layer, but then adds every row to the total. Localizations of other layers are therefore explained by that model too. Continuous and image models do mask by layer.
4. **The free-rotation conversions are inconsistent.** When `xrot`, `yrot` and `zrot` are all free with infinite bounds (the 3D default), the fit switches to an axis-angle vector:
   - `@LocMoFit/LocMoFit.m:468` passes the starting angles to `rotAng2rotMat` in degrees, but that function takes radians. This only matters for non-zero starting angles.
   - After the fit (`LocMoFit.m:501-502`), the rotation is decomposed in the ZYX convention (`rotMat2Ang`, `'rotationMatrixRev'`), while the angles are applied in the XYZ convention (`@LocMoFit/locsHandler.m:111`). The reported Euler angles therefore do not reproduce the fitted rotation; in the numerical check, matrix elements were off by up to about 0.6.
   - Only model 1's own points use the axis-angle vector. Models k > 1 apply model 1's parameters as Euler angles (`modelHandler.m:37`).
5. **Continuous and image models count the blur twice.** `variation` is added in quadrature at `@LocMoFit/locsHandler.m:34` and again at `:200`, so the effective blur is √2 × `variation`.
6. **The cascade does not reset its σ offset.** `sigmaFactor(2)`, the extra σ offset of a cascade step, stays set after the cascade; `@LocMoFit/intensityCal.m:142-144` restores only `sigmaFactor(1)`.
7. **A single model leaves a flat direction.** The model `weight` is free by default, but weights are normalized among the models of a layer. With one model it has no effect, and the optimizer still searches along it.
8. **Two NPC ellipse models have misaligned bounds.** In `dualEllipse3D_discrete` and `dualEllipse3D_avgR_discrete`, `min`/`max` are shifted against the parameter names: `ellipticity` has max 100, `aDir` has max 0.999.
9. **`hemisphere2DPro` clips its own defaults.** The `min`/`max` of both radii are [−10, 10], which excludes their default values of 20 and 40.
10. **`csplineTube3D_xyz` has coincident default control points**: x1 = x2 = 150.

### Issues that change reported diagnostics

11. **Confidence intervals are √2 too wide, and probably not computed at all.**
    - `@LocMoFit/fit.m:331-335` fills H with the coefficients of a fitted quadratic: a_ii on the diagonal, a_ij/2 off it. That is half the Hessian of the quadratic, so the standard deviations are √2 too large.
    - Before that, the code calls a `hessian` function that is not in the SMAP tree, so the computation likely fails and leaves `CI = []`.
    - The error handler reads `obj.linkedGUI.site.indList`, which fails outside the GUI.
12. **The expected log-likelihood (`getLLExpDist`, `LocMoFit.m:1523-1591`) has three problems.**
    - It draws the simulated points with `randi`, ignoring point weights.
    - It reads the z fields, so it works in 3D only.
    - It overwrites its result for each model, so only the last model's result survives.
13. **Fits that stop at the iteration limit are not flagged.** `fit.m` keeps only `[parBestFit, mLLfit]` and discards the solver's exit flag.

### Conventions that differ between models (not bugs, but not to be copied)

- In 2D, `rotcoord2` rotates by −`zrot`; in 3D, a positive `zrot` rotates counterclockwise.
- The 2D point path in `modelHandler` ignores `xscale`/`yscale`.
- `theta` is the full central angle in `arc2D` but the half-angle in `arc2D_arcLen`.
- `csplineTube3D_xyz` and `continousRibbon` space their points in nm, not in units of the precision.
- The `hemispheroid2D` docstring swaps the meanings of `a` and `b`.
- `ellipse2D`/`ellipse3D` sample evenly in parametric angle rather than arc length, so their point density is uneven without weights.

### GUI, documentation and tooling

- `@fitViewer` and `@LocMoFitTest` are empty stubs, and there are no tests. `LocMoFit_manager.batchFit` is empty.
- "Pick site" is described in `docs/basics/GUIOverview.md` but not implemented.
- Several documentation pages are empty, or referenced but missing: `basics/geometricModel`, `basics/parTable`, `programming/buildModel`.
- `template/discreteModel.m` does not run as is. Line 44 calls `parametricModel()`, and the code reads `par.surfaceArea` and `par.closeAngle` while defining `parameter1`/`parameter2`.
- `rstGeneration/generateImgs.m` uses undefined variables and hard-coded Windows paths.
- The particle-fusion scripts (`script/particleFusion.m` and the template-free variants) point to `models/NPCPointModel_flexible2.m`, which exists only in `models/obsolete/`.
- Several obsolete model files declare a class name that differs from their file name.
- Missing compiled binaries: `ndfun` ships a Windows build only, and it is used at `@LocMoFit/locsHandler.m:178`. `ba_interp3` has no Apple-silicon build.
