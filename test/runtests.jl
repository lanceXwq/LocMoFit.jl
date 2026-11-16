using LocMoFit
using LocMoFit:
    Enzyme,
    loss,
    fit_problem,
    ParameterMap,
    free_parameters,
    full_parameters,
    default_parameters,
    unit_points,
    unit_model,
    model_points,
    log_likelihood,
    likelihood_pass,
    exp_nonpositive,
    chart_rotation,
    rotmat
using LinearAlgebra: norm, dot, cross, normalize
using Statistics: mean
using Test
include("tutorial.jl")      # start_params, and the tutorial sites for fit.jl

# Fixed test inputs (no RNG): a deterministic sequence in (-1, 1).
pseudo(i, j) = sin(12.9898i + 78.233j) * 43758.5453 % 1

# Independent geometry (Rodrigues' formula): the unit vector at polar angle θ and azimuth ϕ, the rotation by θ about (-sin ϕ, cos ϕ, 0) that turns z to it, the minimal rotation turning z to a unit vector m, and the pose of an axis with chart coordinates (vx, vy) around a starting axis (θ₀, ϕ₀): that frame, times the minimal rotation to normalize((vx, vy, 1)) in it.
direction(θ, ϕ) = [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
function rodrigues(k, α)
    K = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0]
    return [1.0 0 0; 0 1 0; 0 0 1] + sin(α) * K + (1 - cos(α)) * K * K
end
rotation(θ, ϕ) = rodrigues([-sin(ϕ), cos(ϕ), 0.0], θ)
function minrot(m)
    a = cross([0.0, 0, 1], m)
    return if norm(a) < 1e-15
        rodrigues([1.0, 0, 0], 0.0)
    else
        rodrigues(a / norm(a), acos(m[3]))
    end
end
pose(start, vx, vy) = rotation(start...) * minrot([vx, vy, 1] / norm([vx, vy, 1]))

# Parameters away from the defaults, and each component's starting axis (θ₀, ϕ₀), for every model.
const CASES = (
    (
        "Ring",
        Ring(),
        (x=3.0, y=-2.0, z=5.0, r=40.0, vx=0.1, vy=-0.05, sigma=2.0),
        ((0.4, 1.1),),
    ),
    (
        "SphericalCap",
        SphericalCap(),
        (x=3.0, y=-2.0, z=5.0, s=35.0, alpha=1.1, vx=0.1, vy=-0.05, sigma=2.0),
        ((0.4, 1.1),),
    ),
    (
        "DoubleRing",
        DoubleRing(),
        (
            x1=3.0,
            y1=-2.0,
            z1=5.0,
            r1=50.0,
            vx1=0.1,
            vy1=-0.05,
            x2=4.0,
            y2=-3.0,
            z2=45.0,
            r2=55.0,
            vx2=0.05,
            vy2=0.08,
            sigma=2.0,
        ),
        ((0.4, 1.1), (0.2, -0.7)),
    ),
    (
        "SpCapRing",
        SpCapRing(),
        (
            x1=3.0,
            y1=-2.0,
            z1=5.0,
            r1=40.0,
            vx1=0.1,
            vy1=-0.05,
            x2=4.0,
            y2=-3.0,
            z2=85.0,
            s2=25.0,
            alpha2=1.0,
            vx2=0.05,
            vy2=0.08,
            sigma=2.0,
        ),
        ((0.4, 1.1), (0.2, -0.7)),
    ),
)
const SPACING = 4.0
const FOV = 300.0           # side of the field of view (nm)
# the fractions, away from their defaults: signal fraction logistic(1.2) ≈ 0.77, and composites' shares
fractions(model) = model isa Composite ? (logit_prob=1.2, omega2=0.4) : (logit_prob=1.2,)
# the starting axes as run_locmofit takes them: theta, phi, or thetak, phik for component k
function axes_init(model, starts)
    return if model isa Composite
        merge(
            (
                NamedTuple{(Symbol(:theta, k), Symbol(:phi, k))}(starts[k]) for
                k in eachindex(starts)
            )...,
        )
    else
        (theta=starts[1][1], phi=starts[1][2])
    end
end

# localizations near the model points, and their precisions
function test_data(model, θ, starts; D=3, every=7)
    Y = model_points(model, unit_model(model, θ, starts; spacing=SPACING), θ)[
        :, 1:every:end
    ]
    locs = [Y[d, n] + 6 * pseudo(d, n) for d in 1:D, n in axes(Y, 2)]
    precs = [3 + 4 * abs(pseudo(d + 10, n)) for d in 1:D, n in axes(Y, 2)]
    return locs, precs
end
# the parameters without the chart coordinates vx, vy (vx1, vy1, …), which params cannot give: an axis starts at its chart's center
without_chart(θ) = NamedTuple(k => v for (k, v) in pairs(θ) if first(string(k)) != 'v')
# θ with the fractions and the starting axes as params (start_params: each quantity at its value in θ), held (`fixed`) or with `priors` as named
function case_params(model, θ, starts; background=true, kwargs...)
    f = if background
        fractions(model)
    else
        Base.structdiff(fractions(model), NamedTuple{(:logit_prob,)})
    end
    return start_params(merge(without_chart(θ), f, axes_init(model, starts)); kwargs...)
end
# the problem of a fit at θ (`params` added to or replacing case_params), and its free parameters at θ, chart coordinates included
function problem(
    model,
    θ,
    starts,
    locs,
    precs;
    background=true,
    fixed=(),
    priors=(;),
    params=(;),
    spacing=SPACING,
    fov=FOV,
    kwargs...,
)
    p, u = fit_problem(
        model,
        locs,
        precs;
        params=merge(case_params(model, θ, starts; background, fixed, priors), params),
        spacing,
        fov_volume=fov^size(locs, 1),
        background,
        kwargs...,
    )
    chart = NamedTuple(k => v for (k, v) in pairs(θ) if first(string(k)) == 'v')
    return p,
    free_parameters(merge(full_parameters(u, p.parameter_map), chart), p.parameter_map)
end

# The likelihood of derivations/derivation.qmd with plain loops: per localization, the mean of the Gaussian densities over each component's points (variances precision² + σ²), the components weighted by w, mixed with a uniform background of density 1/V at signal fraction f; minus the log posterior with the log prior `logprior`, per localization.
function reference_loss(locs, precs, Ys, w, σ; f=nothing, V=1.0, logprior=0.0)
    D, N = size(locs)
    total = 0.0
    for n in 1:N
        v = [precs[d, n]^2 + σ^2 for d in 1:D]
        density(y) =
            prod(exp(-(locs[d, n] - y[d])^2 / (2v[d])) / sqrt(2π * v[d]) for d in 1:D)
        ps = sum(
            w[k] * mean(density(Y[:, m]) for m in axes(Y, 2)) for (k, Y) in enumerate(Ys)
        )
        total += log(f === nothing ? ps : f * ps + (1 - f) / V)
    end
    return -(total + logprior) / N
end

# The default priors as run_locmofit documents them: a log-normal most probable at 50 nm (median 50e nm), with logarithmic spread 1, on every ring's radius and cap's scale, Beta(5, 2) on the signal fraction, Dirichlet(5) on a composite's shares, and flat otherwise.
function default_priors(model, background)
    return merge(
        NamedTuple(k => LogNormal(log(50) + 1, 1) for k in names_of(model, (:r, :s))),
        background ? (signal=Beta(5, 2),) : (;),
        model isa Composite ? (shares=Dirichlet(5),) : (;),
    )
end
# the names of a model's quantities among `of` (a ring's :r, a cap's :s and :kappa), with the component's number for a composite
names_of(::Ring, of) = filter(in(of), (:r,))
names_of(::SphericalCap, of) = filter(in(of), (:s, :kappa))
function names_of(c::Composite, of)
    return Tuple(
        Iterators.flatten(
            map(n -> Symbol(n, k), names_of(m, of)) for (k, m) in enumerate(c.components)
        ),
    )
end
# Every kind of prior that is flat by default, on the case's model: a normal density on the height (z, or z2 in component 1's frame), on κ′ and, centered at 0, on the blur; a log-normal on the sizes, and on every axis a Watson density or, on a composite's first axis, a Bingham density (on the second, a girdle).
function every_prior(model)
    composite = model isa Composite
    return merge(
        (composite ? (z2=Normal(70, 15),) : (z=Normal(2, 4),)),
        NamedTuple(k => LogNormal(log(40), 0.5) for k in names_of(model, (:r, :s))),
        NamedTuple(k => Normal(0.3, 0.2) for k in names_of(model, (:kappa,))),
        (sigma=Normal(0, 10),),
        if composite
            (
                axis1=Bingham(Watson(0.5, 1.0, 3.0), Watson(0.5 + π / 2, 1.0, -2.0)),
                axis2=Watson(0.1, -0.3, -2.0),
            )
        else
            (axis=Watson(0.5, 1.0, 3.0),)
        end,
    )
end
# The log prior of derivations/derivation.qmd (Priors on every parameter) with plain formulas, for the priors `priors` at the parameters θ: each density at its quantity, a size's magnitude, a cap's κ′ = sign(s)·sin(α/2), an axis by the independent construction `pose` (for component 2, in component 1's frame), the signal fraction and the shares.
function reference_log_prior(model, θ, starts, priors)
    total = 0.0
    for (name, prior) in pairs(priors)
        s = string(name)
        q = if name == :signal
            1 / (1 + exp(-θ.logit_prob))
        elseif name == :shares
            shares(model, θ)
        elseif startswith(s, "axis")
            k = s[5:end]
            pose(starts[k == "" ? 1 : parse(Int, k)], θ[Symbol(:vx, k)], θ[Symbol(:vy, k)])[
                :, 3
            ]
        elseif startswith(s, "kappa")
            sign(θ[Symbol(:s, s[6:end])]) * sin(θ[Symbol(:alpha, s[6:end])] / 2)
        elseif name == :sigma || s[1] in ('r', 's')
            abs(θ[name])
        else
            θ[name]
        end
        total += if prior isa Normal
            -(q - prior.mean)^2 / (2prior.sd^2)
        elseif prior isa LogNormal
            -log(q) - (log(q) - prior.mu)^2 / (2prior.sigma^2)
        elseif prior isa Watson
            prior.kappa * dot(direction(prior.theta, prior.phi), q)^2
        elseif prior isa Bingham
            sum(t.kappa * dot(direction(t.theta, t.phi), q)^2 for t in prior.terms)
        elseif prior isa Beta
            (prior.a - 1) * log(q) + (prior.b - 1) * log(1 - q)
        elseif prior isa Dirichlet
            (prior.alpha - 1) * sum(log, q)
        else
            0.0
        end
    end
    return total
end
# each component's points (first D rows), and the components' shares, from the parameters
parts(p, θ, D) = [model_points(p.model, p.unit_model, θ)[1:D, r] for r in p.ranges]
function shares(model, θ)
    ω = if model isa Composite
        [0.0; [θ[Symbol(:omega, k)] for k in 2:length(model.components)]]
    else
        [0.0]
    end
    return exp.(ω) ./ sum(exp.(ω))
end

@testset "log-likelihood kernel" begin
    x = [10 * pseudo(i, j) for i in 1:3, j in 1:10]
    y = [10 * pseudo(i, j + 50) for i in 1:3, j in 1:20]
    s = [0.5 + abs(pseudo(i, j + 99)) for i in 1:3, j in 1:10]
    σ = 1.5
    # far from the origin (40 µm), single precision keeps each localization's log density: rounding the coordinates to Float32 moves each difference by at most u, one unit in the last place at 40 µm (0.004 nm), so the log density of each Gaussian errs by at most Σ_d (2|x_d - y_d|·u + u²)/(2v_d) (derivations/derivation.qmd), and the log of their mean by at most the largest of these, a few hundredths here; expanding the square would err by about ε·x²/v ≈ 20
    logp(T, shift) = [
        log_likelihood(
            T.(x[:, n:n] .+ shift),
            T.(s[:, n:n]),
            T.(y .+ shift),
            (1:20,),
            T(σ),
            (zero(T),),
            zero(T),
            T(-Inf),
        ) for n in 1:10
    ]
    shift = 40_000.0
    u = eps(Float32(shift))
    bound = [
        maximum(
            sum((2abs(x[d, n] - y[d, m]) * u + u^2) / (2(s[d, n]^2 + σ^2)) for d in 1:3) for
            m in 1:20
        ) + 1e-4 for n in 1:10
    ]
    @test all(abs.(logp(Float32, shift) .- logp(Float64, 0.0)) .≤ bound)

    # the exponential the kernel uses is within one unit in the last place of the exact value, over the whole range where it is not subnormal; below that it is 0
    for (T, lo) in ((Float64, -708.0), (Float32, -87.0))
        xs = [T(lo * ((1 + pseudo(k, 7)) / 2)^2) for k in 1:20_000]
        @test all(x -> abs(exp_nonpositive(x) - T(exp(big(x)))) ≤ eps(T(exp(big(x)))), xs)
        @test exp_nonpositive(zero(T)) == 1 &&
            exp_nonpositive(T(-1000)) == 0 &&
            exp_nonpositive(T(-Inf)) == 0
    end

    # the gradient computed along with the value, against central differences: with respect to the points, σ, the log weights of two components and the two log fractions of the mixture with the background, in 3D and 2D (where the points' third coordinate has no effect)
    ranges, logw, a, b = (1:8, 9:20), (log(0.3), log(0.7)), log(0.8), log(0.2) - 3log(30.0)
    for D in (3, 2)
        f(y, σ, logw, a, b) = log_likelihood(x[1:D, :], s[1:D, :], y, ranges, σ, logw, a, b)
        ℓ, g = likelihood_pass(x[1:D, :], s[1:D, :], y, ranges, σ, logw, a, b, Val(true))
        @test ℓ ≈ f(y, σ, logw, a, b)
        h = 1e-6
        e(d, m) = [i == d && j == m ? h : 0.0 for i in 1:3, j in 1:20]
        @test g.points ≈ [
            (f(y + e(d, m), σ, logw, a, b) - f(y - e(d, m), σ, logw, a, b)) / 2h for
            d in 1:3, m in 1:20
        ] rtol = 1e-6
        D == 2 && @test all(iszero, g.points[3, :])
        @test g.σ ≈ (f(y, σ + h, logw, a, b) - f(y, σ - h, logw, a, b)) / 2h rtol = 1e-6
        @test g.logw[1] ≈
            (f(y, σ, logw .+ (h, 0), a, b) - f(y, σ, logw .- (h, 0), a, b)) / 2h rtol = 1e-6
        @test g.logw[2] ≈
            (f(y, σ, logw .+ (0, h), a, b) - f(y, σ, logw .- (0, h), a, b)) / 2h rtol = 1e-6
        @test g.a ≈ (f(y, σ, logw, a + h, b) - f(y, σ, logw, a - h, b)) / 2h rtol = 1e-6
        @test g.b ≈ (f(y, σ, logw, a, b + h) - f(y, σ, logw, a, b - h)) / 2h rtol = 1e-6
    end
end

@testset "orientation chart" begin
    # at its center the chart gives the starting axis; elsewhere the pose of the independent construction above, also for a starting axis pointing down
    for start in ((0.4, 1.1), (0.0, 0.0), (Float64(π), 0.3), (2.0, -2.5)),
        (vx, vy) in ((0.0, 0.0), (0.3, -0.2), (-1.5, 2.0))

        R = chart_rotation(rotmat(start...), vx, vy)
        @test R ≈ pose(start, vx, vy)
        @test R' * R ≈ [1.0 0 0; 0 1 0; 0 0 1]
    end
    @test chart_rotation(rotmat(0.4, 1.1), 0.0, 0.0)[:, 3] ≈ direction(0.4, 1.1)
end

@testset "model points" begin
    (_, _, θ, starts) = CASES[1]
    X = model_points(Ring(), unit_model(Ring(), θ, starts; spacing=SPACING), θ)
    c, n = [θ.x, θ.y, θ.z], pose(starts[1], θ.vx, θ.vy)[:, 3]
    @test size(X, 2) == ceil(Int, 2π * θ.r / SPACING)
    @test all(≈(θ.r), norm.(eachcol(X .- c)))
    @test all(x -> abs(dot(x - c, n)) < 1e-9, eachcol(X))

    # the constant-area cap: on the sphere of radius s/κ′ through the apex, within the half-angle acos(1 - 2κ′²) of the axis; its points are uniform in depth below the apex, which on a sphere (the area of a zone is 2πR times its height) means uniform in area; bending the other way for κ′ < 0; and flat at κ′ = 0, with radius 2s
    for α in (1.1, -0.8)
        (_, _, θ, starts) = CASES[2]
        θ = merge(θ, (alpha=α,))
        X = model_points(
            SphericalCap(), unit_model(SphericalCap(), θ, starts; spacing=SPACING), θ
        )
        κ, apex, n = sin(θ.alpha / 2), [θ.x, θ.y, θ.z], pose(starts[1], θ.vx, θ.vy)[:, 3]
        R = θ.s / κ
        center = apex - R * n
        @test size(X, 2) == ceil(Int, 4π * θ.s^2 / SPACING^2)
        @test all(≈(abs(R)), norm.(eachcol(X .- center)))
        @test all(x -> 1 - 2κ^2 - 1e-9 ≤ dot(x - center, n) / R ≤ 1 + 1e-9, eachcol(X))
        depth = [dot(apex - x, n) * sign(κ) for x in eachcol(X)] ./ (2θ.s * abs(κ))      # 0 at the apex, 1 at the rim
        @test all(q -> abs(mean(depth .≤ q) - q) < 2 / size(X, 2), (0.25, 0.5, 0.75))
    end
    (_, _, θ, starts) = CASES[2]
    θ = merge(θ, (alpha=0.0,))
    X = model_points(
        SphericalCap(), unit_model(SphericalCap(), θ, starts; spacing=SPACING), θ
    )
    apex, n = [θ.x, θ.y, θ.z], pose(starts[1], θ.vx, θ.vy)[:, 3]
    @test all(x -> abs(dot(x - apex, n)) < 1e-9 && norm(x - apex) ≤ 2θ.s + 1e-9, eachcol(X))

    # ring 2 sits in ring 1's frame: centered at (x2, y2, z2), radius r2, its axis posed in that frame
    (_, _, θ, starts) = CASES[3]
    U = unit_model(DoubleRing(), θ, starts; spacing=SPACING)
    X = model_points(DoubleRing(), U, θ)
    n1 = size(U[1].points, 2)
    R1 = pose(starts[1], θ.vx1, θ.vy1)
    c1, c2 = [θ.x1, θ.y1, θ.z1], [θ.x1, θ.y1, θ.z1] + R1 * [θ.x2, θ.y2, θ.z2]
    a1, a2 = R1[:, 3], R1 * pose(starts[2], θ.vx2, θ.vy2)[:, 3]
    @test all(≈(θ.r1), norm.(eachcol(X[:, 1:n1] .- c1)))
    @test all(x -> abs(dot(x - c1, a1)) < 1e-9, eachcol(X[:, 1:n1]))
    @test all(≈(θ.r2), norm.(eachcol(X[:, (n1 + 1):end] .- c2)))
    @test all(x -> abs(dot(x - c2, a2)) < 1e-9, eachcol(X[:, (n1 + 1):end]))

    # the cap of SpCapRing: its apex at (x2, y2, z2) in the ring's frame, on the sphere of radius s2/κ′2
    (_, _, θ, starts) = CASES[4]
    U = unit_model(SpCapRing(), θ, starts; spacing=SPACING)
    X = model_points(SpCapRing(), U, θ)
    n1 = size(U[1].points, 2)
    R1 = pose(starts[1], θ.vx1, θ.vy1)
    c1 = [θ.x1, θ.y1, θ.z1]
    a2, ρ = R1 * pose(starts[2], θ.vx2, θ.vy2)[:, 3], θ.s2 / sin(θ.alpha2 / 2)
    center = c1 + R1 * [θ.x2, θ.y2, θ.z2] - ρ * a2
    @test all(≈(θ.r1), norm.(eachcol(X[:, 1:n1] .- c1)))
    @test all(≈(ρ), norm.(eachcol(X[:, (n1 + 1):end] .- center)))
    @test all(
        x -> 1 - 2sin(θ.alpha2 / 2)^2 - 1e-9 ≤ dot(x - center, a2) / ρ ≤ 1 + 1e-9,
        eachcol(X[:, (n1 + 1):end]),
    )
end

@testset "loss against a reference" begin
    # the posterior with its default priors, then with each default replaced (the signal fraction's, the shares', and flat sizes) and with every other quantity given a prior; each distinct variant once (the signal fraction's need a background, the shares' a composite); the log prior does not depend on D, so in 2D only the defaults
    for (_, model, θ, starts) in CASES, D in (3, 2), background in (true, false)
        composite = model isa Composite
        variants = if D == 2
            ((;),)
        else
            unique((
                (;),
                (background ? (signal=Beta(197, 50),) : (;)),
                (background ? (signal=Flat(),) : (;)),
                (composite ? (shares=Dirichlet(2.5),) : (;)),
                (composite ? (shares=Flat(),) : (;)),
                NamedTuple(k => Flat() for k in names_of(model, (:r, :s))),
                every_prior(model),
            ))
        end
        locs, precs = test_data(model, θ, starts; D)
        for priors in variants
            p, u = problem(model, θ, starts, locs, precs; background, priors)
            θp = full_parameters(u, p.parameter_map)
            used = merge(default_priors(model, background), priors)
            @test loss(u, p) ≈ reference_loss(
                locs,
                precs,
                parts(p, θp, D),
                shares(model, θp),
                θp.sigma;
                f=background ? 1 / (1 + exp(-θp.logit_prob)) : nothing,
                V=FOV^D,
                logprior=reference_log_prior(model, θp, starts, used),
            ) rtol = 1e-10
        end
    end
end

@testset "the share prior's optimum" begin
    # two rings 2 µm apart along their axis, n₁ and n₂ localizations exactly on them, no background: every localization comes from its own ring (q = 0 or 1 in derivations/derivation.qmd, @eq-shares), so the share that maximizes the posterior is (n₂ + α - 1)/(n + 2(α - 1)) (@eq-share-map), n₂/n for the flat Dirichlet(1), whose term is 0 (Dirichlet(1) rather than Flat() reuses the compiled loss). The fits also check what a fit without a background returns
    n1, n2, r = 30, 10, 50.0
    ring(n, z) = reduce(
        hcat,
        [[r * cos(2π * (j - 0.5) / n), r * sin(2π * (j - 0.5) / n), z] for j in 1:n],
    )
    locs = hcat(ring(n1, 0.0), ring(n2, 2000.0))
    precs = fill(5.0, size(locs))
    θ = (
        x1=0.0,
        y1=0.0,
        z1=0.0,
        r1=r,
        theta1=0.0,
        phi1=0.0,
        x2=0.0,
        y2=0.0,
        z2=2000.0,
        r2=r,
        theta2=0.0,
        phi2=0.0,
        sigma=1.0,
    )
    for α in (5.0, 2.0, 1.0)
        fit = run_locmofit(
            DoubleRing(),
            locs,
            precs;
            params=merge(
                start_params(θ; fixed=keys(start_params(θ))),
                (shares=ParamToFit(Dirichlet(α)),),
            ),
            background=false,
            tol=1e-10,
        )
        @test fit.weights[2] ≈ (n2 + α - 1) / (n1 + n2 + 2(α - 1)) rtol = 1e-6
        @test !haskey(fit.parameters, :logit_prob) &&
            fit.signal_fraction === nothing &&
            isfinite(fit.loss)
    end
end

@testset "changing the length unit" begin
    # every length divided by L (positions, radii, scales, the blur, the spacing, and the log-normal priors' medians): the posterior's geometry is the same, and the loss (minus the log posterior per localization) drops by (D + J/N)·log L exactly, with J log-normal priors (derivations/derivation.qmd, @eq-units-priors)
    L = 2.5
    scaled(θ) = NamedTuple{keys(θ)}(
        Tuple(
            first(string(k)) in ('x', 'y', 'z', 'r', 's') ? v / L : v for
            (k, v) in pairs(θ)
        ),
    )
    for (_, model, θ, starts) in CASES, D in (3, 2)
        locs, precs = test_data(model, θ, starts; D)
        p, u = problem(model, θ, starts, locs, precs)
        sizes = names_of(model, (:r, :s))
        pL, uL = problem(
            model,
            scaled(θ),
            starts,
            locs ./ L,
            precs ./ L;
            priors=NamedTuple(k => LogNormal(log(50 / L) + 1, 1) for k in sizes),
            spacing=SPACING / L,
            fov=FOV / L,
        )
        N = size(locs, 2)
        @test loss(uL, pL) ≈ loss(u, p) - (D + length(sizes) / N) * log(L) rtol = 1e-10
    end
end

@testset "priors" begin
    # where the likelihood does not depend on a quantity, the fit ends at its prior's mode (derivations/derivation.qmd, Priors on every parameter): the height of a ring in 2D under a normal density; and the cap of a SpCapRing whose share is held at 10⁻²⁰, fitted to a ring's localizations with a background, so that the cap's density is negligible next to the background's: its height z2 at the normal's mean, its scale at the log-normal's mode exp(μ - σ²), its κ′ at the normal's mean, and its axis along ±d under a Bingham density that holds it near the plane perpendicular to ν = (θ, ϕ) = (0.5 + π/2, 1) (κ = −5) and pulls it toward d = (0.5, 1), which is perpendicular to ν, within that plane (κ = 2): d is the eigenvector of A with the largest eigenvalue, 2, against 0 and −5
    (_, model, θ, starts) = CASES[1]
    locs, precs = test_data(model, θ, starts; D=2)
    held = Tuple(k for k in keys(case_params(model, θ, starts)) if k != :z)
    fit = run_locmofit(
        model,
        locs,
        precs;
        params=case_params(model, θ, starts; fixed=held, priors=(z=Normal(30, 10),)),
        spacing=SPACING,
        tol=1e-10,
    )
    @test fit.parameters.z ≈ 30 rtol = 1e-8

    ring = (x1=3.0, y1=-2.0, z1=5.0, r1=40.0, theta1=0.4, phi1=1.1)
    locs, precs = test_data(
        Ring(), (x=3.0, y=-2.0, z=5.0, r=40.0, vx=0.0, vy=0.0, sigma=2.0), ((0.4, 1.1),)
    )
    start = merge(
        ring,
        (x2=0.0, y2=0.0, z2=85.0, s2=30.0, alpha2=π / 3, theta2=0.2, phi2=-0.7),
        (sigma=2.0, logit_prob=1.2),
    )
    free = (:z2, :s2, :kappa2, :axis2)
    params = merge(
        start_params(
            start; fixed=Tuple(k for k in keys(start_params(start)) if !(k in free))
        ),
        (shares=ParamToFit((1.0, 1e-20), Fixed()),),
    )
    priors = (
        z2=Normal(70, 5),
        s2=LogNormal(log(40), 0.5),
        kappa2=Normal(0.3, 0.2),
        axis2=Bingham(Watson(0.5 + π / 2, 1.0, -5.0), Watson(0.5, 1.0, 2.0)),
    )
    params = merge(
        params, NamedTuple(k => ParamToFit(params[k].value, priors[k]) for k in free)
    )
    fit = run_locmofit(SpCapRing(), locs, precs; params, spacing=SPACING, tol=1e-10)
    θf = fit.parameters
    @test θf.z2 ≈ 70 rtol = 1e-8
    @test abs(θf.s2) ≈ 40exp(-0.5^2) rtol = 1e-8
    @test sign(θf.s2) * sin(θf.alpha2 / 2) ≈ 0.3 rtol = 1e-8
    @test abs(dot(pose((0.2, -0.7), θf.vx2, θf.vy2)[:, 3], direction(0.5, 1.0))) ≈ 1 rtol =
        1e-10

    # what params accepts
    locs, precs = test_data(model, θ, starts)
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; params=(r2=ParamToFit(1.0),)
    )       # not a quantity of the model
    @test_throws ArgumentError fit_problem(Ring(), locs, precs; params=(r=50.0,))                  # not a ParamToFit
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; params=(x=ParamToFit(LogNormal(0, 1)),)
    )     # a log-normal fits sizes only
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; params=(r=ParamToFit(Watson(0, 0, 1)),)
    )    # a Watson density fits axes only
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; params=(r=ParamToFit(Bingham(Watson(0, 0, 1))),)
    )    # and so does a Bingham density
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; params=(axis=ParamToFit(Normal(0, 1)),)
    )
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; params=(signal=ParamToFit(Dirichlet(5)),)
    )
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; background=false, params=(signal=ParamToFit(0.5),)
    )   # no signal fraction without a background
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; params=(signal=ParamToFit(1.0),)
    )  # outside (0, 1)
    @test_throws ArgumentError fit_problem(
        SphericalCap(), locs, precs; params=(kappa=ParamToFit(1.0),)
    )
    @test_throws ArgumentError fit_problem(
        Ring(), locs, precs; params=(axis=ParamToFit(0.3),)
    )   # an axis is (theta, phi)
    @test_throws ArgumentError fit_problem(
        DoubleRing(), locs, precs; params=(shares=ParamToFit((0.5, 0.3, 0.2)),)
    )
    @test_throws ArgumentError fit_problem(
        DoubleRing(), locs, precs; params=(r2=ParamToFit(Tied(:z1)),)
    )   # a size tied to a position
    @test_throws ArgumentError fit_problem(
        DoubleRing(), locs, precs; params=(r2=ParamToFit(Tied(:r3)),)
    )
    @test_throws ArgumentError fit_problem(
        DoubleRing(), locs, precs; params=(axis2=ParamToFit(Tied(:axis1)),)
    )
    @test_throws ArgumentError Normal(0, 0)
    @test_throws ArgumentError LogNormal(0, -1)
    @test_throws ArgumentError Beta(0.5, 2)
    @test_throws ArgumentError Dirichlet(0.5)
    @test_throws ArgumentError Bingham()
end

@testset "composites" begin
    # names: each component's own names with its number appended, one shared blur, and the weights
    c = Composite(SphericalCap(), Ring(); defaults=(z2=30,))
    θ = default_parameters(c, Float64)
    @test keys(θ) == (
        :x1,
        :y1,
        :z1,
        :s1,
        :alpha1,
        :vx1,
        :vy1,
        :x2,
        :y2,
        :z2,
        :r2,
        :vx2,
        :vy2,
        :sigma,
        :omega2,
    )
    @test θ.s1 == default_parameters(SphericalCap(), Float64).s &&
        θ.r2 == 50.0 &&
        θ.z2 == 30.0 &&
        θ.omega2 == 0
    @test keys(default_parameters(Composite(Ring(), Ring(), Ring()), Float64))[(end - 2):end] ==
        (:sigma, :omega2, :omega3)
    # with component 1 at the origin and unrotated, each component's points are its own model's points
    θ = merge(θ, (vx2=0.3, vy2=-0.1, x2=5.0))
    starts = ((0.0, 0.0), (0.5, 1.0))
    U = unit_model(c, θ, starts; spacing=SPACING)
    X = model_points(c, U, θ)
    cap_alone = model_points(
        SphericalCap(), U[1], (x=0.0, y=0.0, z=0.0, s=θ.s1, alpha=θ.alpha1, vx=0.0, vy=0.0)
    )
    ring_alone = model_points(Ring(), U[2], (x=5.0, y=0.0, z=30.0, r=θ.r2, vx=0.3, vy=-0.1))
    @test X ≈ hcat(cap_alone, ring_alone)
    # the world-frame summary of each component
    comps = LocMoFit.components(c, U, θ)
    @test comps[2].position ≈ [5.0, 0.0, 30.0] &&
        comps[2].axis ≈ pose(starts[2], 0.3, -0.1)[:, 3] &&
        comps[2].r == 50.0
    @test comps[1].kappa ≈ sin(θ.alpha1 / 2) &&
        comps[1].radius ≈ θ.s1 / sin(θ.alpha1 / 2) &&
        comps[1].area ≈ 4π * θ.s1^2
end

@testset "freezing" begin
    (_, model, θ, starts) = CASES[1]
    locs, precs = test_data(model, θ, starts)
    m = ParameterMap(θ, (:z, :sigma))
    u = free_parameters(θ, m)
    @test propertynames(u) == (:x, :y, :r, :vx, :vy)
    @test full_parameters(u, m) == θ
    u[:r] = 45.0
    @test full_parameters(u, m) == merge(θ, (r=45.0,))
    pf, uf = problem(model, θ, starts, locs, precs; fixed=(:z, :sigma))
    uf[:r] = 45.0
    @test loss(uf, pf) ≈ loss(
        problem(model, merge(θ, (r=45.0,)), starts, locs, precs)[2],
        problem(model, θ, starts, locs, precs)[1],
    )
    @test_throws ArgumentError ParameterMap(θ, (:w,))
    @test_throws ArgumentError ParameterMap(θ, keys(θ))
end

@testset "ties" begin
    (_, model, θ) = CASES[3]
    m = ParameterMap(θ, (:sigma,), (:r2 => :r1,))
    u = free_parameters(θ, m)
    @test !(:r2 in propertynames(u)) && !(:sigma in propertynames(u))
    @test full_parameters(u, m) == merge(θ, (r2=θ.r1,))        # the target takes the source's value
    u[:r1] = 47.0
    @test full_parameters(u, m).r2 == 47.0
    # tied to a frozen parameter: frozen at its value; ties chain
    mf = ParameterMap(θ, (:r1,), (:r2 => :r1,))
    @test full_parameters(free_parameters(θ, mf), mf).r2 == θ.r1
    mc = ParameterMap(θ, (), (:r2 => :r1, :z2 => :r2))
    @test full_parameters(free_parameters(θ, mc), mc).z2 == θ.r1
    @test_throws ArgumentError ParameterMap(θ, (), (:r2 => :w,))             # unknown
    @test_throws ArgumentError ParameterMap(θ, (), (:r2 => :r1, :r2 => :x1))  # tied twice
    @test_throws ArgumentError ParameterMap(θ, (:r2,), (:r2 => :r1,))         # frozen and tied
    @test_throws ArgumentError ParameterMap(θ, (), (:r2 => :r1, :r1 => :r2))  # a cycle
end

@testset "gradient" begin
    # Enzyme's gradient of the loss (through the rule for the log-likelihood) against central differences, for every model with the background and the weights: Ring with the default priors; SphericalCap and SpCapRing with the default priors and every other kind of prior; DoubleRing with its radii tied and its second axis held, and once without a background; SpCapRing also in 2D
    for (_, model, θ, starts, kwargs, D) in (
        (CASES[1]..., (;), 3),
        (CASES[3]..., (fixed=(:axis2,), params=(r2=ParamToFit(Tied(:r1)),)), 3),
        (CASES[3]..., (background=false,), 3),
        (CASES[4]..., (;), 2),
        (CASES[2]..., (priors=every_prior(CASES[2][2]),), 3),
        (CASES[4]..., (priors=every_prior(CASES[4][2]),), 3),
    )
        locs, precs = test_data(model, θ, starts; D)
        p, u = problem(model, θ, starts, locs, precs; kwargs...)
        du = zero(u)
        Enzyme.autodiff(
            Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(u, du), Enzyme.Const(p)
        )
        fd = map(eachindex(u)) do i
            h = 1e-6 * max(1, abs(u[i]))
            up, um = copy(u), copy(u)
            up[i] += h
            um[i] -= h
            (loss(up, p) - loss(um, p)) / 2h
        end
        @test du ≈ fd rtol = 1e-5
    end
end

@testset "run_locmofit" begin
    (_, model, θ, starts) = CASES[1]
    locs, precs = test_data(model, θ, starts; every=3)
    init = merge(
        without_chart(θ), (x=0.0, z=0.0, r=30.0, sigma=1.0), axes_init(model, starts)
    )
    r = run_locmofit(
        "Ring", locs, precs; params=start_params(init; fixed=(:sigma,)), spacing=SPACING
    )
    @test r.history === nothing
    @test r.parameters.sigma == 1.0                                     # frozen
    @test 0 < r.signal_fraction < 1 &&
        r.signal_fraction ≈ 1 / (1 + exp(-r.parameters.logit_prob))
    @test r.weights == (1.0,)
    @test r.components[1].axis ≈
        chart_rotation(r.unit_model.chart, r.parameters.vx, r.parameters.vy)[:, 3]
    @test r.components[1].axis ≈ direction(r.components[1].theta, r.components[1].phi)
    p0, u0 = fit_problem(
        Ring(), locs, precs; params=start_params(init; fixed=(:sigma,)), spacing=SPACING
    )
    @test r.loss < loss(u0, p0)
    @test r.unit_model.points == unit_points(model, init; spacing=SPACING)   # not changed by the fit
    @test r.model_points ≈ model_points(model, r.unit_model, r.parameters)

    h = run_locmofit(
        "Ring",
        locs,
        precs;
        params=start_params(init; fixed=(:sigma,)),
        spacing=SPACING,
        history=true,
    )
    @test h.parameters == r.parameters
    @test length(h.history.losses) == length(h.history.parameters) > 1
    @test all(q -> q.sigma == 1.0, h.history.parameters)

    # a tie holds throughout, and the unit points are built with it; a composite can be passed as a model, and its shares sum to one
    (_, model, θ, starts) = CASES[3]
    locs, precs = test_data(model, θ, starts)
    t = run_locmofit(
        Composite(Ring(), Ring()),
        locs,
        precs;
        params=merge(case_params(model, θ, starts), (r2=ParamToFit(Tied(:r1)),)),
        spacing=SPACING,
        max_iter=20,
    )
    @test t.parameters.r2 == t.parameters.r1
    @test size(t.unit_model[2].points) == size(t.unit_model[1].points)
    @test sum(t.weights) ≈ 1 && length(t.weights) == 2
end

@testset "point_params" begin
    # the number of points comes from point_params (ceil(2πr/spacing) for a ring, ceil(4πs²/spacing²) for a cap); the starting values from params, held or not
    (_, model, θ, starts) = CASES[1]
    locs, precs = test_data(model, θ, starts; every=3)
    npoints(p) =
        map(u -> size(u.points, 2), p.unit_model isa Tuple ? p.unit_model : (p.unit_model,))
    p80, u80 = fit_problem(Ring(), locs, precs; params=(r=ParamToFit(80.0),), spacing=3.0)
    p50, u50 = fit_problem(
        Ring(),
        locs,
        precs;
        params=(r=ParamToFit(80.0),),
        point_params=(r=50.0,),
        spacing=3.0,
    )
    @test npoints(p80) == (ceil(Int, 2π * 80 / 3),)
    @test npoints(p50) == (ceil(Int, 2π * 50 / 3),)
    @test u50 == u80 && u50.r == 80
    pf, uf = fit_problem(
        Ring(),
        locs,
        precs;
        params=(r=ParamToFit(80.0, Fixed()),),
        point_params=(r=50.0,),
        spacing=3.0,
    )
    @test npoints(pf) == (ceil(Int, 2π * 50 / 3),) &&
        full_parameters(uf, pf.parameter_map).r == 80
    pc, _ = fit_problem(SphericalCap(), locs, precs; point_params=(s=30.0,), spacing=3.0)
    @test npoints(pc) == (ceil(Int, 4π * 30^2 / 3^2),)
    # with a tie, the target's points follow the source's value
    pd, ud = fit_problem(
        DoubleRing(),
        locs,
        precs;
        params=(r1=ParamToFit(70.0), r2=ParamToFit(Tied(:r1))),
        point_params=(r1=40.0,),
        spacing=3.0,
    )
    @test npoints(pd) == (ceil(Int, 2π * 40 / 3), ceil(Int, 2π * 40 / 3)) && ud.r1 == 70
end

@testset "starting values from the data" begin
    # derivations/derivation.qmd, Starting values from the data: noise-free points with isotropic precisions, so that the expected values follow from the geometry
    estimate_start = LocMoFit.estimate_start
    n, c = direction(0.4, 1.1), [3.0, -2.0, 5.0]
    R = rotation(0.4, 1.1)                          # turns z to n
    angle_of(e, k="") = direction(e[Symbol(:theta, k)], e[Symbol(:phi, k)])

    # a ring: its center is the mean, its axis the eigenvector with the smallest eigenvalue, its radius every point's distance from the axis
    ring = R * (40 .* LocMoFit.unit_ring(Float64, 120)) .+ c
    e = estimate_start(Ring(), ring, fill(2.0, size(ring)), nothing)
    @test [e.x, e.y, e.z] ≈ c && e.r ≈ 40 && angle_of(e) ≈ n

    # a cap, each point with its mirror image through the axis: the axis is exact and the apex lies on it; flat, the apex is the center and s² is the true s² less half the summed precision variance (here 3·1²), since the mean squared distance from the apex is 2s²
    U = LocMoFit.unit_cap(Float64, 400)
    U = hcat(U, [-U[1:2, :]; U[3:3, :]])
    for α in (0.0, 1.0)
        cap = R * LocMoFit.local_points(SphericalCap(), U, (s=30.0, alpha=α)) .+ c
        e = estimate_start(SphericalCap(), cap, fill(1.0, size(cap)), nothing)
        @test angle_of(e) ≈ n                       # κ′ ≥ 0, as the true cap's
        @test norm(cross([e.x, e.y, e.z] - c, n)) < 1e-9
        α == 0 &&
            @test [e.x, e.y, e.z] ≈ c && e.s ≈ sqrt(30^2 - 3 / 2) && abs(e.alpha) < 1e-9
    end

    # the two scores against loops over the derivation's formulas, on scattered localizations
    locs = [
        10 * pseudo(d, i) + (d == 3 ? 30 * sign(pseudo(9, i)) : 0) for d in 1:3, i in 1:40
    ]
    precs = [2 + abs(pseudo(d + 20, i)) for d in 1:3, i in 1:40]
    f = LocMoFit.site_frame(locs, precs)
    s = LocMoFit.AxisScores(f)
    m = normalize([0.2, -0.1, 1.0])
    LocMoFit.score!(s, (Ring(), Ring()), m)
    G = s.order[1:(s.nlow)]
    w = [dot(f.rel[:, i], m) for i in G]
    ρ = [sqrt(norm(f.rel[:, i])^2 - dot(f.rel[:, i], m)^2) for i in G]
    along = [sum(f.var[:, i] .* m .^ 2) for i in G]
    across = [(sum(f.var[:, i]) - a) / 2 for (i, a) in zip(G, along)]
    cw = f.c[G]
    mean_w(x) = sum(cw .* x) / sum(cw)
    @test LocMoFit.part_score(Ring(), s, G) ≈
        sum(cw .* ((ρ .- mean_w(ρ)) .^ 2 .+ (w .- mean_w(w)) .^ 2 .- across .- along))
    A = sqrt.(cw) .* [ones(length(G)) ρ .^ 2]
    ab = A \ (sqrt.(cw) .* w)
    @test LocMoFit.part_score(SphericalCap(), s, G) ≈
        sum(cw .* ((w .- ab[1] .- ab[2] .* ρ .^ 2) .^ 2 .- along))

    # two coaxial rings far apart compared with their radius (r = 35, d = 65 nm), along one of the coarse candidate axes: there both groups are circles around it, so its score is the lowest and the search returns it exactly, and each ring's start is exact
    a = LocMoFit.hemisphere(LocMoFit.START_COARSE)[:, 30]
    Ra = minrot(a)
    unit = 35 .* LocMoFit.unit_ring(Float64, 60)
    two = hcat(Ra * unit .+ (c - 32.5a), Ra * unit .+ (c + 32.5a))
    e = estimate_start(DoubleRing(), two, fill(2.0, size(two)), nothing)
    @test angle_of(e, 1) ≈ a
    @test [e.x1, e.y1, e.z1] ≈ c - 32.5a && e.r1 ≈ 35 && e.r2 ≈ 35
    @test e.z2 ≈ 65 && abs(e.omega2) < 1e-12

    # fit_problem: given values are kept, held ones not given keep their defaults, a tie's target follows its source, and the rest is estimated
    p, u = fit_problem(
        DoubleRing(),
        two,
        fill(2.0, size(two));
        params=(r1=ParamToFit(50.0), z1=ParamToFit(Fixed()), r2=ParamToFit(Tied(:r1))),
    )
    θ = full_parameters(u, p.parameter_map)
    @test θ.r1 == 50 && θ.r2 == 50 && θ.z1 == 0
    @test [θ.x1, θ.y1] ≈ (c - 32.5a)[1:2] && θ.z2 ≈ 65
    @test p.unit_model[1].chart[:, 3] ≈ a
    # a starting axis given is used
    p, _ = fit_problem(
        DoubleRing(), two, fill(2.0, size(two)); params=(axis1=ParamToFit((0.0, 0.0)),)
    )
    @test p.unit_model[1].chart[:, 3] ≈ [0, 0, 1]
    # in 2D only the position is estimated
    p, u = fit_problem(Ring(), ring[1:2, :], fill(2.0, 2, size(ring, 2)))
    θ = full_parameters(u, p.parameter_map)
    @test [θ.x, θ.y] ≈ c[1:2] &&
        θ.z == 0 &&
        θ.r == 50 &&
        p.unit_model.chart[:, 3] ≈ [0, 0, 1]
    # start_names lists what estimate_start gives, so that fit_problem can skip it when all of them are given
    for m in (
            Ring(),
            SphericalCap(),
            DoubleRing(),
            SpCapRing(),
            Composite(SphericalCap(), Ring()),
            Composite(Ring(), Ring(), Ring()),
        ),
        D in (3, 2)

        @test keys(estimate_start(m, locs[1:D, :], precs[1:D, :], nothing)) ==
            LocMoFit.start_names(m, D)
    end
    # a model without an estimate of its shape starts from its defaults, but with component 1 at the localizations' weighted center: here c, about which the two rings are symmetric
    p, u = fit_problem(Composite(Ring(), Ring(), Ring()), two, fill(2.0, size(two)))
    θ = full_parameters(u, p.parameter_map)
    @test [θ.x1, θ.y1, θ.z1] ≈ c
    @test Base.structdiff(θ, NamedTuple{(:x1, :y1, :z1)}) == Base.structdiff(
        merge(default_parameters(p.model, Float64), (logit_prob=log(9),)),
        NamedTuple{(:x1, :y1, :z1)},
    )
end

include("fit.jl")
