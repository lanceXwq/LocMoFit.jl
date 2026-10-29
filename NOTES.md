# Implementation notes

Why the fit is implemented the way it is, with the measurements behind each choice. The math is in `derivations/derivation.qmd`; how to use and change the code is in `AGENTS.md`. All timings are on an Apple M1 Pro (8 performance and 2 efficiency cores), Julia 1.13.3, on the tutorial sites of `examples/data` from the moved start of `test/fit.jl`; the scripts are in `benchmarks/speed/` and its dated result files.

## Speed of the likelihood (2026-10-27)

The likelihood (`derivations/derivation.qmd`) needs, for every localization, a log-sum-exp over every model point, so a fit costs time proportional to N·M, the number of localization–point pairs. Before these changes, `loss` built the M×N matrix of log densities with broadcasts, and Enzyme differentiated through them:

| | value | gradient |
| --- | --- | --- |
| broadcasts + Enzyme (before) | 6.5 ns per pair | 25 ns per pair |
| the loop with its rule, Float64 | 1.7 ns | 2.7 ns |
| the loop with its rule, Float32 | 0.7 ns | 1.2 ns |

Complete fits became 5–25× faster: a cap of 290 localizations and 6,648 points from 3.4 s to 0.29 s (Float64) and from 4.4 s to 0.18 s (Float32), a ring from 19 ms to 2.8 ms (`before_rule_2026-10-27.txt`, `rule_2026-10-27.txt`). The results did not change: losses agree with the broadcasts to 10⁻¹⁴ and gradients to 10⁻¹⁵ (relative) in Float64, Float32 to about one unit in the last place, and the accuracy benchmark's quick subset gave the same bias and scatter to every printed digit. What made the difference, in order of gain:

1. **A loop with a hand-written Enzyme rule** (`log_likelihood`, `likelihood_pass`, `src/rules.jl`). For each localization the loop keeps the terms of its points in a buffer of M numbers; once the localization's density is known, one more pass over the buffer gives the derivatives with respect to the points and the variances (`derivations/derivation.qmd`, the gradient of the log-likelihood). The gradient therefore costs about 1.7 values and needs no N×M memory. Enzyme differentiates only what lies around the loop (parameters → model points, weights, background, prior, penalty), which is O(M + N). This is DynMoFit's idea (its `cap_logsumexp` rule), but simpler here: all localizations share the same model points, so the points are placed once per evaluation and the inner loop is a diagonal Gaussian, with no per-pair bend or rotation.
2. **A vectorizable exponential**, `exp_nonpositive`. `Base.exp` looks up a table, so a loop over it does not vectorize; after the other changes it was most of a value's cost. `exp_nonpositive` reduces x = k·log 2 + r, evaluates the Taylor polynomial of exp(r) (degree 13 in Float64, 7 in Float32) and builds 2ᵏ from the exponent bits. It is within one unit in the last place of the exact value (tested), and in a summing loop it took 1.0 ns instead of 3.4 ns (Float64) and 0.38 ns instead of 3.6 ns (Float32). With 14 coefficients `evalpoly` is no longer inlined by itself, and the call alone made it 2.8× slower; it is inlined by hand.
3. **Two passes instead of a running maximum.** A streaming log-sum-exp (rescale whenever a new maximum appears) has a branch and does not vectorize. The loop first finds the smallest exponent with `@fastmath min` (plain `min` or `ifelse` did not vectorize: 1.24 against 0.47 ns per point), then sums the terms.
4. **`@inbounds` inside every closure.** `@inbounds` on a loop does not reach into the `ntuple` closures inside it, and the remaining bounds checks stopped vectorization: the per-point kernels ran 2–4× slower until each closure got its own `@inbounds`. Bounds were checked separately: the threaded quick subset with `--check-bounds=yes` raised no error.
5. **The coordinates as contiguous vectors.** The loop copies each row of the 3×M points into its own vector (and accumulates the gradient likewise). With views of the matrix rows instead, a value and a gradient were 1.2–2.4× slower (measured 2026-10-28).

**The value with every gradient** (`objective`). BFGS's backtracking line search asks for the value at a trial point and, once it accepts the point (nearly always the first one), for the gradient there, which through Optimization.jl (a `TwiceDifferentiable` objective) was a second, separate evaluation. `objective` computes every value together with its gradient (Enzyme's `ReverseWithPrimal`) and keeps the gradient for the call that follows. Fits got 1.1–1.6× faster with bit-identical results; extra line-search steps cost a gradient instead of a value, which is why Float32, with more backtracking, gains less.

**Allocations.** A gradient allocates O(M) arrays (the model points, Enzyme's shadows, the loop's buffers), not O(N·M). Writing the cap's `local_points` as one fused broadcast halved what a cap gradient allocates (2.8 → 1.6 MB) and made it 7% faster. Fusing `place` the same way allocated less but made the gradient slower, so it was not kept.

**Tried and not kept:** a streaming (single-pass) log-sum-exp; views instead of contiguous copies; a fused `place`; and computing distances in the model's frame, which pays off in DynMoFit only because every localization there has its own model.

## Parallel fits

Each fit runs on one thread. Fitting many sites in `Threads.@threads` ran 80 cap fits 5.7× faster on 8 threads than one after another, the rest being garbage collection, which takes 16% of a threaded run. But threaded fits have crashed Julia three times with memory corruption: twice with the broadcast loss ("GC error (probable corruption)", 2026-10-26) and once with the rule (2026-10-27, malloc detected a corrupted heap after 22 of the 25 groups of a full accuracy run). Five cold-start threaded runs and the threaded quick subset with bounds checks did not crash. The cause is not known; it is not an out-of-bounds write in the loop. Separate processes would avoid it, but the owner chose to keep the code simple (2026-10-27), so the benchmarks run their full set on one thread (2 minutes).

Threads within one fit (splitting the localizations inside the rule, where Enzyme does not see them) would help a single large site, as `fitfunction` does in DynMoFit (7× on 8 threads); it is not implemented.

## Reproducibility

Float64 fits are bit-identical across sessions and thread counts. Float32 fits are reproducible only for a fixed `julia -t`: the loss is the same, but Enzyme's compiled gradient differs in its last bits with the thread count, and BFGS then takes a different path to the same optimum (the tutorial cap: 44 or 52 iterations, final losses 1·10⁻⁴ apart, far below the statistical error). This was so before the rule as well, where the paths parted at iteration 2 instead of 5.

## Known issue: Float32 fits can stop early (found 2026-10-28, open)

In Float32 (the default `dtype` until 2026-10-29, now an option), BFGS can stop before the gradient reaches `tol`, and the fit still reports `Success`. Optim.jl also stops when the loss is unchanged in two successive iterations (|f − f′| ≤ `f_reltol`·|f| with `f_reltol = 0`), and near the optimum a Float32 loss stops changing. Found while porting LocMoFitPy, whose float32 fits of the same tutorial sites reach the float64 optimum:

| Tutorial site (moved start) | Float32 iterations (Float64) | largest gradient component at the stop | loss above the Float64 optimum, per localization | largest parameter error |
| --- | --- | --- | --- | --- |
| ring | 34 (43) | 6.6e-4 | 8e-5 | z 0.25 nm |
| spherical cap | 42 (50) | 4.6e-4 | 1.6e-4 | apex x 0.30 nm, σ 0.52 nm |
| cap with ring | 63 (94) | 6.5e-4 | 3.0e-3 | the cap's apex 3.3 and 6.4 nm in x2, y2 |
| ring in 2D, double ring | 54, 94 (53, 87) | — | none (they reach the optimum) | — |

**Effect on accuracy.** Over the benchmark's 20 sites per model (`benchmarks/accuracy/prior_5_2_2026-10-28.txt`, background, moved start), the Float32 groups have the same bias as the Float64 groups and a scatter within about 10% of theirs (the cap with ring's cap center: 2.2–12.4 nm against 2.2–11.1 nm), with fewer iterations (65 against 92). So the early stops add error that is small against the statistical error but not zero, and they are not a bias.

**Likely cause.** The kernel sums the localizations' log-densities one after another in Float32 (`likelihood_pass`), so the total, about −5,000 for a site, carries rounding errors of order 10⁻⁵ per localization that do not change smoothly with the parameters; XLA sums them pairwise, which is why LocMoFitPy's float32 fits are not affected. Not yet confirmed by a fix.

**Proposed fix (not done).** Combine each localization's terms and sum them over localizations in Float64 inside the kernel, keeping the work per point in Float32 (it costs nothing measurable: the sums are per localization), and add a test that a Float32 fit of the tutorial data reaches the gradient tolerance. Float64 became the default on 2026-10-29 (the owner's decision), so the issue affects only fits that ask for Float32.

## The signal prior (2026-10-28)

The Beta prior on the signal fraction π keeps its mode at 0.8 (the owner's choice) but became flatter: Beta(5, 2), worth 5 localizations, instead of DynMoFit's Beta(197, 50), worth 245. On the benchmark's sites with 15% background, Beta(197, 50) pulled π by −0.025 to −0.033 (z down to −13); Beta(41, 11) by about −0.01 to −0.016; Beta(17, 5) by −0.004 to −0.009; Beta(5, 2) by at most 0.004, within its scatter of about 0.013. The geometry was the same under every prior. On data without background the fitted π went from 0.85–0.90 to 0.985–0.994 (truth 1). Beta(5, 2) still vanishes at π = 0 and 1, so π stays inside (0, 1) (`benchmarks/accuracy/prior_5_2_2026-10-28.txt`).
