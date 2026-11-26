# To-do

## Major

### Accuracy

1. **Torus vs ring.** Test whether a torus fits better than the ring. The ring with the fitted blur σ is already a torus with a Gaussian tube cross-section, so the test is an explicit tube radius (points on the torus surface, or in its volume) against that.
2. **Localizations outside the selected ROI.** The background density is uniform over the ROI's volume V, so localizations beyond V (and signal cut off at the ROI's edge) break its normalization. Simulate the case with DynMoSim and measure the bias.
3. **Axis guess from the membrane side.** Sites sit on the cell membrane: the interior side has more localizations (signal and background), the exterior is mostly dark. Use that asymmetry to guess the starting axis and its sign.
4. **Which way a cap bends.** A directional (von Mises–Fisher) prior on the axis. For a SpCapRing it needs a positive prior on `z2` too, since the likelihood is invariant under (n, z2, κ′) → (−n, −z2, −κ′).
5. **SpCapRing: the cap is often smaller than the ring.** Use it to improve the start, or as a prior on the ratio of the cap's size to the ring's.
6. **Float32 fits far from the origin.** Cap fits at 100 µm stop early, up to 2.8× their statistical error from the optimum (`DESIGN.md`). Candidates: center the localizations on the site inside `run_locmofit` (and add the center back on output), or fixed-point numbers where Float32 loses resolution (coordinates in integer units of 2⁻ᵏ nm, or an integer accumulator for the loss sum). Measure Float32 at 100 µm against the origin and against Float64. Mirror the fix in LocMoFitPy.

### Performance

1. **Threaded fits of many sites crash rarely** (memory corruption, cause unknown; `AGENTS.md`, Many sites). Find the cause, so that sites can be fitted in threads (5.7× faster on 8 threads).

### Simplicity

### Extensibility

1. **Cylinder with a shell thickness** as a model.
2. **Model points in a volume** once a model is 3D (torus, cylinder shell). The point count then grows with the volume, so check the speed and the `spacing` rule.
3. **Start from the original LocMoFit's fits.** A helper script (not package code) that reads MATLAB LocMoFit's fitted parameters (for example from an `_sml.mat`) and turns them into starting values in LocMoFit.jl's parametrization.

## Minor

1. The size prior's spread, and per-model defaults.
2. A proper prior for the blur.
3. 2D starts: along a given axis (`z2`, the ring's position); a 2D ring started untilted stays untilted, since the projection is even in the tilt.
4. The prior types' names clash with Distributions.jl's.
5. A helper to choose `spacing`.
6. Features of MATLAB LocMoFit still open (several channels, scale, point weights, passing parameters between fits, cascades, AIC, the expected log-likelihood, confidence intervals): see `MATLAB_DIFFERENCES.md`.
