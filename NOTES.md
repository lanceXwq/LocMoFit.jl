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

## Known issue: Float32 fits can stop early (found 2026-10-28, accepted 2026-10-29)

In Float32 (the default `dtype` until 2026-10-29, now an option), BFGS can stop before the gradient reaches `tol`, and the fit still reports `Success`. Optim.jl also stops when the loss is unchanged in two successive iterations (|f − f′| ≤ `f_reltol`·|f| with `f_reltol = 0`), and near the optimum a Float32 loss stops changing. Found while porting LocMoFitPy, whose float32 fits later showed the same early stops:

| Tutorial site (moved start) | Float32 iterations (Float64) | largest gradient component at the stop | loss above the Float64 optimum, per localization | largest parameter error |
| --- | --- | --- | --- | --- |
| ring | 34 (43) | 6.6e-4 | 8e-5 | z 0.25 nm |
| spherical cap | 42 (50) | 4.6e-4 | 1.6e-4 | apex x 0.30 nm, σ 0.52 nm |
| cap with ring | 63 (94) | 6.5e-4 | 3.0e-3 | the cap's apex 3.3 and 6.4 nm in x2, y2 |
| ring in 2D, double ring | 54, 94 (53, 87) | — | none (they reach the optimum) | — |

**Effect on accuracy.** Over the benchmark's 20 sites per model (`benchmarks/accuracy/prior_5_2_2026-10-28.txt`, background, moved start), the Float32 groups have the same bias as the Float64 groups and a scatter within about 10% of theirs (the cap with ring's cap center: 2.2–12.4 nm against 2.2–11.1 nm), with fewer iterations (65 against 92). So the early stops add error that is small against the statistical error but not zero, and they are not a bias.

**Cause.** Near the optimum the decreases BFGS still needs are smaller than what Float32 resolves in the loss: the mean loss, about 15, is resolved to about 10⁻⁶, and so is each localization's log density. It is not how the kernel sums the localizations: LocMoFitPy sums them pairwise (XLA) and its float32 fits stop early as often (its `NOTES.md`), and there, reverting each of its speed optimizations in turn (the closed-form gradient, evaluating the value with the gradient, padding) changed the share of early stops only within the noise of the starting point.

**Accepted (2026-10-29, the owner's decision).** Float64 is the default and does not stop early. A Float32 fit is faster (a cap 1.6×) and can stop with the loss up to about 10⁻⁴ per localization above the optimum and the parameters off by up to about 0.5–0.9 (nm or the cap's units) on the tutorial sites, far inside their statistical error but not zero; use Float64 where that matters.

**Far from the origin it is worse (measured 2026-11-13).** SMLM coordinates are absolute, up to about 100–200 µm on a large camera, and in Float32 a coordinate at 100 µm is resolved only to 0.0078 nm (at 100 nm, to 8·10⁻⁶ nm). The accuracy benchmark's sites (20 per model, background, moved start) were fitted with x and y moved by 25 and 100 µm and compared with the Float64 fit at the original position (scratch scripts; the same in LocMoFitPy, its `NOTES.md`). Each cell: the largest change of the position and of the axis over the sites, in units of the statistical error (the RMS error of the Float64 fits against the truth), and the median loss above the Float64 optimum, per localization:

| | Float32 at the origin | at 25 µm | at 100 µm |
| --- | --- | --- | --- |
| Ring | 0.08, 0.02; 3·10⁻⁶ | 0.04, 0.02; 4·10⁻⁷ | 0.14, 0.04; 8·10⁻⁶ |
| SphericalCap | 0.92, 0.86; 3·10⁻⁵ | 0.45, 0.43; 6·10⁻⁵ | 2.8, 2.7; 1.2·10⁻² |
| DoubleRing | 0.17, 0.18; 8·10⁻⁵ | 0.69, 0.59; 1·10⁻⁴ | 0.34, 0.70; 1·10⁻⁴ |
| SpCapRing | 0.60, 1.3; 1.4·10⁻³ | 0.48, 1.2; 8·10⁻⁴ | 0.60, 1.2; 1.5·10⁻³ |

Float64 fits moved by 100 µm changed by at most 6·10⁻⁷ nm. At 100 µm every Float32 cap fit stopped on an unchanged loss (14 of the 20 after a last iteration that moved no parameter at all), after a median of 30 iterations (44 at the origin), up to 13.6 nm and 6.9° from the optimum, and every one reported `Success`. The model points are placed at the absolute coordinates, so a parameter change that moves them by less than about 0.004 nm is rounded away, and the loss changes only in steps on that scale; once the line search's steps are that small, the loss stops changing and Optim.jl stops. Most likely the cap is hit hardest because it approaches its optimum slowly, in small steps along which its apex, size and curvature trade off, so it reaches that scale far from the optimum. The other models stay within their statistical error at every offset.

**Possible remedies (not done).** (1) Finish a Float32 fit with a few Float64 iterations from where it stopped: exact, and it keeps most of the speed, since the Float64 part starts near the optimum. (2) Subtract each localization's log density at the start from the loss, so that the loss is near 0 and its changes keep more digits: in LocMoFitPy, on 10 starts nudged around each tutorial site, it cut the early stops of the ring, the cap and the cap with ring from 0, 6 and 9 to 0, 1 and 3 out of 10, so it helps but does not end them. (3) Subtract the site's position from the localizations before the fit and add it back to the fitted positions, as SMAP's LocMoFit does (which also computes in double precision): it would remove the loss of accuracy far from the origin, though not the early stops at the origin. Any of them would be mirrored in LocMoFitPy.

## The signal prior (2026-10-28)

The Beta prior on the signal fraction π keeps its mode at 0.8 (the owner's choice) but became flatter: Beta(5, 2), worth 5 localizations, instead of DynMoFit's Beta(197, 50), worth 245. On the benchmark's sites with 15% background, Beta(197, 50) pulled π by −0.025 to −0.033 (z down to −13); Beta(41, 11) by about −0.01 to −0.016; Beta(17, 5) by −0.004 to −0.009; Beta(5, 2) by at most 0.004, within its scatter of about 0.013. The geometry was the same under every prior. On data without background the fitted π went from 0.85–0.90 to 0.985–0.994 (truth 1). Beta(5, 2) still vanishes at π = 0 and 1, so π stays inside (0, 1) (`benchmarks/accuracy/prior_5_2_2026-10-28.txt`).

## The share prior (2026-11-14)

**Why.** On real data a composite lost a component. On the owner's first real dataset (20 late endocytic sites of yeast, 2D, side view; fitted in a scratch study with LocMoFitPy as a coaxial SpCapRing, the axis in the image plane and the cap held as a half-sphere of 40 nm), the cap's share fell to 0 on 10 of the 20 sites: started below its cluster of localizations, the cap first lost its share (on one site from 0.5 to 0.001 within 20 iterations), after which the loss no longer depended on where the cap was (flat to 1e-5 per localization over 200 nm of z2), and the fit ended with the cap wherever its start had put it. With the cap held at each z2 and the rest refitted from a share of 1/2, that site had a fit 0.36 per localization (42 in log posterior) better. `derivations/derivation.qmd`, A prior on the shares, gives the mechanism and the prior.

**Measured** in the same scratch study, with the prior patched into LocMoFitPy's loss for the run. Each site's fits were scored with the loss without the share prior against the best fit found for it (the lowest of all variants and of a profile over z2 with the share restarted at 1/2); "at the best" is within 0.01 per localization. Scored with each prior's own loss instead, the counts were the same.

| | Components switched off (share < 0.05) | Sites at the best | Shares moved, against free shares where those kept both components |
| --- | --- | --- | --- |
| Free shares | 10 | 9 | — |
| Dirichlet(2) | 1 | 17 | 0.002 |
| Dirichlet(5) | 0 | 18 | 0.007 |
| Dirichlet(10) | 0 | 17 | 0.013 |
| Shares frozen equal (as the original LocMoFit held its weights) | 0 | 9 | 0.073 |
| Free shares, z2 started at the lower cluster | 1 | 18 | 0.000 |

A start near the cap's localizations worked as well as the prior there, but that start depended on the dataset's frame; the prior applies to every composite. The two sites that miss the best with Dirichlet(5) differ in the ring and the background, not the cap: a long ring that calls nearly everything signal, or a shorter one that calls the band's end background. Localizations of a nearby site may explain that (the owner), so they are not counted as failures of the fit.

**Choice.** Dirichlet(5) by default (`share_prior = 5`, the owner's decision, 2026-11-14), passed as `nothing` for none. Values below 1 are refused, since the prior then grows without bound as a share goes to 0.

**Measured on the accuracy benchmark** (`benchmarks/accuracy/share_prior_2026-11-14.txt`, against the same run with `share_prior = nothing`): the geometry did not change (every error within a few percent of its scatter), the simple models were identical, and the double ring's share (true 1/2) kept its bias. The ring's share of the cap with a ring (true 0.3) moved by +0.0064 to +0.0075 (z 2.4–2.7, against a scatter of 0.012 over the sites). For fixed responsibilities, `derivations/derivation.qmd` (@eq-share-map) predicts +0.0052 to +0.0054 for these sites' 265 to 361 signal localizations; the rest comes from the responsibilities, which move with the shares. `test/fit.jl` takes the composites' tolerances from this run and allows that share its bias.

## Priors on every parameter (2026-11-14, branch `priors`, a prototype)

**Why.** The owner's decision (2026-11-14): every quantity a fit estimates has a prior, flat for some for now, and what is known about a quantity is stated as a proper prior, not as a bound or a soft penalty. The share prior was the first such case; the size penalties (sizes ≥ 1 nm) were the last constraints outside the posterior. `derivations/derivation.qmd`, Priors on every parameter, has the math: each prior is a density of a quantity of the structure, added with no Jacobian, so the fit is the most probable value of the quantities whatever coordinates the optimizer uses.

**Design.** `params` holds a `ParamToFit(value, prior)` per quantity, value and prior in the quantity's own terms; four quantities differ from the optimizer's parameters and get their own names (`axis` as `(theta, phi)`, `kappa`, `shares`, `signal`; since 2026-11-17 `shares` and `signal` are one quantity, `fractions`: below). Holding and tying are priors too (`Fixed()`, `Tied(:r1)`), so `init_params`, `freeze`, `tie`, `signal_prior` and `share_prior` are gone. `fit_problem` turns `params` into the existing parameter map (starting values, held parameters, ties) plus a tuple of `PriorTerm`s, whose parameter names are type parameters, so that the loss adds them with plain field access; a Watson density is turned into the chart's coordinates once, so it costs a dot product. The priors' numbers are fields, so changing them does not recompile the loss; changing a family does (a new type, compiled like a first fit). The prior types are LocMoFit's own (no Distributions.jl, a dependency through which Enzyme is untested); `Normal`, `LogNormal`, `Beta` and `Dirichlet` would clash with Distributions.jl's exports in a session that loads both. The defaults are those of the main branch except for the sizes: a log-normal on every ring's radius and cap's scale replaces the penalty, since it vanishes at 0 (`LogNormal(log(50), 1)` at first, `LogNormal(log(50) + 1, 1)` since the same day: below).

**Speed** (`benchmarks/speed/speed.jl`, twice each on the main branch and the prototype, alternating, 8 threads): the loss and its gradient took the same time within ±2% for every model and dtype, and Float64 fits the same time within run-to-run noise. Some Float32 fits took a different number of iterations, since the objective changed (the ring 34 → 41, the double ring 87 → 89, the cap 43 → 41).

**Accuracy** (the quick subset of `benchmarks/accuracy/fit.jl`, the same 20 sites per model on both branches): the log-normal moved every size by −0.012 to −0.035 nm (ring r: bias −0.059 → −0.071; ring in 2D r: 0.474 → 0.439; cap s: −0.149 → −0.174; double ring r1, r2: 0.287 → 0.273, 0.131 → 0.117; the cap with a ring's r: −0.024 → −0.038), at most 5% of their scatter (0.55–1.0 nm); every other error changed by at most a few percent of its scatter. `test/fit.jl`'s tolerances still hold.

**A threaded fit gave a wrong result once.** In the first quick subset on the prototype (8 threads), one of the 20 cap sites did not converge and was far off (the scatter of the apex's z went from 1.73 to 2.13 nm). Rerun with 8 threads and with 1, the group was identical in both, all 20 converged, and site by site the prototype's fits were within 0.03 nm of the fits without the size prior. Float64 fits are otherwise bit-identical across thread counts (Reproducibility), so this looks like the threaded fits' memory corruption (Parallel fits), seen for the first time as a wrong result rather than a crash.

**The size prior's mode (2026-11-14).** A fit maximizes the posterior, so a size prior pulls a size toward its most probable value, exp(μ − σ²), not its median. For `LogNormal(log(50), 1)` that is 18 nm, which pulled every larger size down: on 20 yeast CME sites (real 2D data, coaxial SpCapRing, about 100 localizations each), rings shrank by 0.3–1.3 nm, and caps whose scale was free shrank to 9–15 nm on 5 of 20 sites. The default is now `LogNormal(log(50) + 1, 1)`, most probable at 50 nm and still weak (median 136 nm, 95% between 19 and 965 nm), the owner's choice for now. The quick subset of `benchmarks/accuracy/fit.jl` (the same 20 sites per model, 8 threads, all converged), against the clean rerun with the former default: ring r bias −0.071 → −0.059 nm, ring in 2D r 0.439 → 0.473, cap s −0.174 → −0.153, double ring r1, r2 0.273 → 0.287, 0.117 → 0.131, the cap with a ring's r −0.038 → −0.023; every other error within a few percent of its scatter. The cap's curvature c stays flagged at z = −3.1 (bias −0.000141 → −0.000138), as it was before this change.

**The cap's curvature as its half-angle (2026-11-14).** κ′ = tanh η put the whole sphere at η → ∞, where the slope sech²η vanishes: a fit that came near it could not come back. On the yeast sites with κ′ free (a normal prior around the half-sphere, from one start for all sites), 8 of 20 fits ended at κ′ = 1 with η between 5 and 222 (slopes 10⁻⁵ to 10⁻¹⁹³), 5 of them in clearly worse optima; with κ′ flat, 18 of 20. The cap is now fitted by its half-angle α, κ′ = sin(α/2) (`derivations/derivation.qmd`, @eq-cap-alpha): the whole sphere lies at the finite α = ±π. The same fits, run with a patched copy of LocMoFitPy's branch (identical fits with κ′ held, same iterations on all 20 sites): with the normal prior the 8 stuck fits reached caps with κ′ 0.66–0.81, better by 2.2–12.8 in log posterior, and none got worse; with κ′ flat, closed spheres fell from 18 to 6, where a start from the half-sphere fit also ends. The remaining misses there (yeast ROIs 7 and 12) are starts far from the cap's localizations, not the parametrization. A start at α = ±π would be stationary (every function of κ′ is), so a starting cap is still kept within |κ′| ≤ 0.95. Speed (`benchmarks/speed/speed.jl`, twice): loss and gradient within 1% for every model; some fits take another path (Float32 cap 41 → 45 iterations, Float32 cap with a ring 62 → 51). Accuracy: the quick subset above, run with both changes. The parameter is `alpha` (`alpha2`, … in a composite); the default, 0.961 (55°), keeps the former default κ′ ≈ 0.46.

**A Bingham prior on axes (2026-11-16).** The owner asked for an axis held almost exactly in the image plane whose turn within it stays in a soft range around its start, as the original yeast fits bounded the turn by ±25°. One Watson density has one spread, so `Bingham(terms::Watson...)` adds the exponents of several (`derivations/derivation.qmd`, @eq-bingham): `Bingham(Watson(0, 0, -1000), Watson(π/2, -π/2, 2.6))` holds the tilt to about 1.3° (SD) and the turn around −y′ to about 25°. Watson and Bingham both compile to one quadratic form in the chart's coordinates (`ChartQuadratic`, six numbers), so the loss has one path for both, and changing a κ does not recompile. Tests: the reference loss and the gradient with a Bingham density on a composite's first axis, a fit that ends at a Bingham density's mode, and the argument checks. The test on the yeast sites is LocMoFitPy's (its NOTES.md).

**Axis priors compared (2026-11-17).** The von Mises–Fisher density, κ(μ·n), is directional: it tells n from −n and has one peak around +μ (SD ≈ 1/√κ), with normalization κ/(4π sinh κ). The Watson density, κ(μ·n)², is axial: two peaks ±μ (κ > 0) or a band around the great circle perpendicular to μ (κ < 0), one spread (SD ≈ 1/√(2|κ|)). The Bingham density, nᵀAn with A = Σⱼ κⱼμⱼμⱼᵀ, is axial too and has one spread per eigendirection of A; Watson is its one-term case. All three are proper for every κ, and the fit never needs their normalizations unless κ is fitted. Watson and Bingham are implemented (one quadratic form in the chart); von Mises–Fisher is not (it would be linear in the chart's m). An axis known as a line (a ring's) takes Watson or Bingham; a direction (which side a composite's second part lies on) needs von Mises–Fisher. But SpCapRing's likelihood is unchanged by (n, z2, κ′) → (−n, −z2, −κ′), so a von Mises–Fisher prior on n alone cannot put the cap on one side of the ring; it needs a prior that keeps z2 positive as well, and then κ′'s sign is the cap's bending direction. That is how yeast ROI 15 could be kept from flipping (cap above the ring) under the Bingham prior; not done.

**The background as fraction 0, one Dirichlet prior (2026-11-17).** The owner's choice: the signal fraction and the shares become one mixture, the background as fraction f₀ and the components as f₁, …, f_K = softmax(`omega0`, 0, `omega2`, …) (`derivations/derivation.qmd`, @eq-softmax), with one quantity `fractions` and one prior, `Dirichlet(2, 5, …, 5)` with a background and `Dirichlet(5)` without (@eq-fraction-prior). Offered against keeping a Beta on π and a Dirichlet on the shares, which keeps the two independent; the owner took the single Dirichlet. It equals the former priors for one component; for K components it acts on π as Beta(4K + 1, 2) rather than Beta(5, 2) (@eq-fraction-split: two components pull π toward 8/9 rather than 0.8), and on the shares as the former Dirichlet(5). `Beta`, `logit_prob`, `signal` and `shares` are gone; the kernel lost its signal argument `a` (the components' log fractions already carry π). A background fraction not given starts at 0.1 whatever the components' starting shares. Tutorial fits (`../LocMoFitPy/tests/data/julia_fits.json`): the simple models' are unchanged (ω₀ = −`logit_prob`, the same iterations and loss); the double ring's π 0.872 → 0.875 and the cap with a ring's 0.859 → 0.861, every position within 0.02 nm. Speed (`benchmarks/speed/speed.jl`, twice each against the previous commit, alternating, 8 threads): loss and gradient within about 2% for every model and dtype, Float64 fits the same iterations and time; some Float32 fits took another path (the ring 43 → 30 iterations, the cap 45 → 42, the cap with a ring 51 → 86, 0.04 → 0.07 s). The full accuracy benchmark (`benchmarks/accuracy/fractions_prior_2026-11-17.txt`, against main's `share_prior_2026-11-14.txt`): all 500 fits converged; every geometric error within 0.13 of its scatter; the composites' signal fraction with a background moved toward the truth (double ring −0.0028 → +0.0010, the cap with a ring −0.0034 → −0.0013, scatter about 0.011); no new flag. `test/fit.jl`'s tolerances were taken from it (each within 2% of the former).

**Not ported:** `jlbridge.py`, `examples/fit/`, `benchmarks/start/` and `benchmarks/session/` still use the old keywords; `examples/data/`'s parameter files and `cases.jl` still name a cap's curvature `eta` (κ′ = tanh eta), which `test/tutorial.jl` reads alongside `alpha`.

**Open:** the size prior's spread, and whether one default suits every model (now most probable at 50 nm, spread 1); a prior on the blur (flat and improper; a half-normal, `Normal(0, τ)`, would make it proper); positions, flat and improper; a prior on which way a cap bends, a joint density of its axis and κ′ that the Watson density does not give; an axis held in part, as the yeast fits held `vy1` to keep the axis in the image plane, which now is the Watson girdle `Watson(0, 0, kappa)` with `kappa < 0`, or, with a soft range for its turn too, a Bingham density (below).

## Starting values from the data (2026-11-12)

**Why.** BFGS climbs to the maximum whose basin holds the start, and the posterior has others: for two rings, one at the axis perpendicular to theirs (`derivations/derivation.qmd`, Starting values from the data). From the fixed defaults (the center at the origin, the axis along z, the default shapes) a quarter to all of the simulated sites below ended at another maximum. `fit_problem` therefore starts every parameter that `init_params` leaves open, and that is neither frozen nor a tie's target, from `estimate_start` (`src/start.jl`). It skips the estimate when all the parameters it gives are set, so a fully specified fit (the benchmarks, the moved start of `test/fit.jl`) costs what it did.

**Choices.** Measured first in scratch tests with LocMoFitPy (2026-11-11 and 12), then here:

- The localizations within twice the median distance from their median: that drops most of the background.
- Each weighted by the inverse of its summed precision variance: the precisions are heavy-tailed (dim events), and the weights halved the median axis error of a ring's or a cap's start (7.9° to 3.5°).
- One part: the axis is the covariance's eigenvector with the smallest eigenvalue. Two parts: the eigenvalues are nearly equal (two rings with d ≈ √2 r), so a search scores 100 directions over the hemisphere, then the best one and 64 within 25° of it.
- The ring's score includes the spread of the heights along the axis. Without it, 22 of the 100 coaxial double rings started near the axis perpendicular to theirs and stayed there; with it, none.
- A ring and a cap are told apart by the spread of their distances from the axis relative to the median. The search score cannot do it, since a cap's parabola fits a ring's group as well: assigned by the score, 13 of the 100 coaxial cap + ring sites started with the parts swapped.
- A cap's size is the weighted mean of |x − apex|² − Σ s², over 2. Unweighted, a few dim localizations made it negative on 5 of the cap + ring sites, so s started at the floor of 1 nm, and those fits missed.
- A model without an estimate of its shape (a composite of one part, or of three or more) gets component 1's position at the weighted center of the localizations (2026-11-13). Its other parameters start from the defaults; without the position, it started at the origin, micrometers from a site in real coordinates.
- Not adopted: picking one of several starts by the loss at the start (it picked the wrong one on 7 to 26 of 100 sites), and keeping the best of several fits, which costs 3 to 4 fits per site. It was needed only while the ring's score lacked the heights' spread, and could return as an option.
- 2D: only the position. A 2D ring started untilted stays untilted: its projection is the same tilted either way, so the start is a stationary point of the loss in the tilt (as before, with the defaults).

**Measured** with `benchmarks/start/` (`generate.jl`, then `fit.jl`; `estimate_2026-11-13.txt`) on DynMoSim sites with LocMoFit.jl's benchmark acquisition (shot-noise precisions, 15% background in a 300 nm cube, `fov_volume` = 300³ nm³), every site with its own true parameters:

- 200 rings: r 30–80 nm, 100 signal localizations.
- 100 caps: s 30–60 nm, κ′ 0.1–0.9, 250 signal localizations.
- 100 double rings: equal radii of 35–65 nm, 35–65 nm apart, 160 signal localizations.
- 100 caps above a ring: r 35–65 nm, s 20–35 nm, κ′ 0.3–0.8, the cap's rim 20–50 nm above the ring, 300 signal localizations.

The axis was tilted up to 60° (rings, double rings) or 45° (caps, cap + ring), and the center was up to 30 nm off laterally and 60 nm axially. A fit missed when its loss ended more than 0.01 per localization above that of the fit started at the truth. "Defaults" is the start before this change. Times are the totals of the fits on one thread, after compilation; the estimate's cost is per site.

| | Missed, estimated start | Missed, defaults | Estimate's axis off (median) | Estimate | Fits, estimated start | Fits, defaults |
| --- | --- | --- | --- | --- | --- | --- |
| Ring | 0 | 25 | 3.5° | 0.04 ms | 0.6 s | 1.1 s |
| SphericalCap | 0 | 24 | 3.6° | 0.15 ms | 11.1 s | 20.0 s |
| DoubleRing, coaxial (`x2, y2, vx2, vy2` frozen) | 0 | 58 | 3.3° | 1.0 ms | 1.1 s | 1.8 s |
| DoubleRing, all free | 0 | 70 | 3.3° | 1.0 ms | 1.5 s | 2.0 s |
| SpCapRing, coaxial | 0 | 93 | 6.0° | 2.2 ms | 7.6 s | 38.7 s |
| SpCapRing, all free | 3 | 98 | 6.0° | 2.2 ms | 8.9 s | 40.6 s |

The estimated axis was never more than 45° off. The 3 fits that missed from the estimated start had the second component free, and the same starts did not miss with it held coaxial; their cap's curvature started low (κ′ about 0.1 against 0.5–0.7), since the noise in ρ² flattens the parabola fitted to it.

## Using LocMoFit.jl from SMAPpy (2026-11-13)

**The question.** SMAPpy fits through LocMoFitPy, with the plugin LocMoFitPy ships. Calling LocMoFit.jl instead, through juliacall (`jlbridge.py`), would spare porting each new feature. What would it cost? Not adopted; recorded here for when the question comes back.

**Time and memory**, measured with `benchmarks/session/session.jl` (`session_2026-11-13.txt`) and LocMoFitPy's `benchmarks/speed/session.py` (its `session_2026-11-13.txt`): fresh sessions, each model on its tutorial site, 3D float64 unless noted, Apple M1 Pro, one thread, LocMoFitPy without its persistent compilation cache. "Then" is the median of five fits after the first.

| | LocMoFit.jl | LocMoFitPy |
| --- | --- | --- |
| Load or import | 2.5 s | 1.0 s |
| Ring: first fit, then | 43.0 s, 2.4 ms | 1.6 s, 7.2 ms |
| SphericalCap | 5.4 s, 206 ms | 1.6 s, 339 ms |
| DoubleRing | 6.3 s, 12.4 ms | 1.8 s, 28.3 ms |
| SpCapRing | 5.8 s, 82 ms | 2.0 s, 142 ms |
| Ring 2D, after the 3D models | 0.2 s, 1.7 ms | 1.1 s, 6.2 ms |
| Ring, Float32 | 6.4 s, 3.2 ms | 1.2 s, 4.9 ms |
| All of the above, from the start of the session | 72 s | 13 s |
| Peak memory | 0.97 GB | 0.56 GB |

The first fit of a session compiles, with Enzyme, what all models share, and each later model or element type adds 5–6 s; the 2D ring reuses the 3D ring's code. LocMoFitPy compiles per model and size class, 1–2 s each, and its SMAPpy plugin keeps them on disk across sessions (LocMoFitPy's `NOTES.md`), so a repeat session compiles almost nothing, while LocMoFit.jl pays its 43 s again in every session. The times after the first fit are those of one site and changed by up to 40% between runs (an earlier run: the cap 222 ms here and 248 ms in LocMoFitPy); `benchmarks/speed/` measures them properly. With these, the 43 s are earned back after roughly 300–1,700 caps, 700–850 caps with a ring, 2,700–2,800 double rings or 9,000–10,500 rings (over the two runs), so on datasets of a few hundred sites LocMoFitPy finishes first.

**A precompile workload does not cache the compilation.** A scratch package that depended on LocMoFit.jl and fitted the tutorial ring in a PrecompileTools workload:

```julia
module LocMoFitWarm
using LocMoFit, PrecompileTools
@compile_workload begin
    locs, precs = site(ring_csv)   # the tutorial ring, D×N
    run_locmofit("Ring", locs, precs)
end
end
```

precompiled in 38 s and loaded in 2.0 s, but the first fit in a new session then crashed with a segmentation fault, in both of two runs: Enzyme's compiled code, taken from the package image, does not run. Not tried: a system image (PackageCompiler, a new dependency). It would have to be rebuilt after every update of LocMoFit.jl, while the SMAPpy plugin updates itself from `main`.

**Install.** Julia 1.13 takes 0.9 GB, and LocMoFit.jl's 146 packages (Enzyme, Optimization, SciMLBase, …) about 0.4 GB with their binaries and one precompiled cache each. How long a fresh install takes to precompile them was not measured. juliacall would be a new dependency of SMAPpy's environment; its JuliaPkg can download Julia itself.

**Threads**, from juliacall's documentation, not tested. SMAPpy runs every plugin step in a `QThread`. juliacall can be called from any Python thread since version 0.9.22 (marked experimental) and holds the GIL during every call, so SMAPpy's Python side (the progress line, the Stop button) would wait for each fit and for the whole first compilation, unless the plugin calls through `_jl_call_nogil`, which requires that the Julia code not touch Python. JAX releases the GIL while it computes. With several Julia threads, juliacall recommends `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, since Julia's garbage collector otherwise crashes the process, and that keeps Ctrl-C from reaching Python.

**Not tested:** Julia in SMAPpy's process, next to PySide6 and PyTorch. Julia and JAX ran in one process without trouble (macOS, the check of `jlbridge.py` below).

**`jlbridge.py`** was rewritten on 2026-11-13 to take the arguments of LocMoFitPy's `run_locmofit`, including `point_params` and (N, D) localizations, and to return LocMoFitPy's dict, so that a caller switches by changing the import. Checked in one process against LocMoFitPy (scratch script; juliacall 0.9.37 on LocMoFit.jl's environment, its PythonCall updated to 0.9.37, and on a scratch environment with Enzyme 0.13.219, with the same results): on the five tutorial sites from the estimated start, the SpCapRing site from the moved start, and with `freeze`, `tie`, `point_params`, `fov_volume`, no prior, no background, `max_iter` and `history`, the two dicts had the same keys, types and shapes, and in Float64 the same iterations, return codes and values to 1e-10 (history included); Float32 fits differed by up to 0.25 nm, as Float32 fits of the two packages do (the known issue above). Importing the module took 5–6 s and the first fit 52 s. No test runs it, since CI has no juliacall.

**What it would not save.** MATLAB calls LocMoFitPy (`LocMoFitPy/run_locmofit.m`), so LocMoFitPy, and its parity with LocMoFit.jl, stay unless MATLAB moves too.
