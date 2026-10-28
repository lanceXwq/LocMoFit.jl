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
    penalty,
    log_likelihood,
    likelihood_pass,
    exp_nonpositive,
    chart_rotation,
    rotmat
using LinearAlgebra: norm, dot, cross
using Statistics: mean
using Test

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
        (x=3.0, y=-2.0, z=5.0, s=35.0, eta=0.6, vx=0.1, vy=-0.05, sigma=2.0),
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
            eta2=0.5,
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
function problem(model, θ, starts, locs, precs; background=true, kwargs...)
    f = if background
        fractions(model)
    else
        Base.structdiff(fractions(model), NamedTuple{(:logit_prob,)})
    end
    return fit_problem(
        model,
        locs,
        precs;
        init_params=merge(θ, f, axes_init(model, starts)),
        spacing=SPACING,
        dtype=Float64,
        fov_volume=FOV^size(locs, 1),
        background,
        kwargs...,
    )
end

# The likelihood of derivations/derivation.qmd with plain loops: per localization, the mean of the Gaussian densities over each component's points (variances precision² + σ²), the components weighted by w, mixed with a uniform background of density 1/V at signal fraction f; minus the log posterior with the Beta(a, b) prior, per localization.
function reference_loss(locs, precs, Ys, w, σ; f=nothing, V=1.0, prior=nothing)
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
    logprior = if prior === nothing || f === nothing
        0.0
    else
        (prior[1] - 1) * log(f) + (prior[2] - 1) * log(1 - f)
    end
    return -(total + logprior) / N
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
    for η in (0.6, -0.4)
        (_, _, θ, starts) = CASES[2]
        θ = merge(θ, (eta=η,))
        X = model_points(
            SphericalCap(), unit_model(SphericalCap(), θ, starts; spacing=SPACING), θ
        )
        κ, apex, n = tanh(θ.eta), [θ.x, θ.y, θ.z], pose(starts[1], θ.vx, θ.vy)[:, 3]
        R = θ.s / κ
        center = apex - R * n
        @test size(X, 2) == ceil(Int, 4π * θ.s^2 / SPACING^2)
        @test all(≈(abs(R)), norm.(eachcol(X .- center)))
        @test all(x -> 1 - 2κ^2 - 1e-9 ≤ dot(x - center, n) / R ≤ 1 + 1e-9, eachcol(X))
        depth = [dot(apex - x, n) * sign(κ) for x in eachcol(X)] ./ (2θ.s * abs(κ))      # 0 at the apex, 1 at the rim
        @test all(q -> abs(mean(depth .≤ q) - q) < 2 / size(X, 2), (0.25, 0.5, 0.75))
    end
    (_, _, θ, starts) = CASES[2]
    θ = merge(θ, (eta=0.0,))
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
    a2, ρ = R1 * pose(starts[2], θ.vx2, θ.vy2)[:, 3], θ.s2 / tanh(θ.eta2)
    center = c1 + R1 * [θ.x2, θ.y2, θ.z2] - ρ * a2
    @test all(≈(θ.r1), norm.(eachcol(X[:, 1:n1] .- c1)))
    @test all(≈(ρ), norm.(eachcol(X[:, (n1 + 1):end] .- center)))
    @test all(
        x -> 1 - 2tanh(θ.eta2)^2 - 1e-9 ≤ dot(x - center, a2) / ρ ≤ 1 + 1e-9,
        eachcol(X[:, (n1 + 1):end]),
    )
end

@testset "loss against a reference" begin
    for (_, model, θ, starts) in CASES,
        D in (3, 2), background in (true, false),
        prior in ((197.0, 50.0), (3.0, 2.0), nothing)

        !background && prior !== (197.0, 50.0) && continue          # the prior only matters with a background
        locs, precs = test_data(model, θ, starts; D)
        p, u = problem(model, θ, starts, locs, precs; background, signal_prior=prior)
        θp = full_parameters(u, p.parameter_map)
        @test penalty(model, θp) == 0
        f = background ? 1 / (1 + exp(-θp.logit_prob)) : nothing
        @test loss(u, p) ≈ reference_loss(
            locs, precs, parts(p, θp, D), shares(model, θp), θp.sigma; f, V=FOV^D, prior
        ) rtol = 1e-10
    end
end

@testset "changing the length unit" begin
    # every length divided by L (positions, radii, scales, the blur, the spacing): the posterior's geometry is the same, and the loss (minus the log posterior per localization) drops by D·log L exactly
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
        pL, uL = fit_problem(
            model,
            locs ./ L,
            precs ./ L;
            init_params=merge(scaled(θ), fractions(model), axes_init(model, starts)),
            spacing=SPACING / L,
            dtype=Float64,
            fov_volume=(FOV / L)^D,
        )
        @test loss(uL, pL) ≈ loss(u, p) - D * log(L) rtol = 1e-10
    end
end

@testset "penalties" begin
    # soft bounds on sizes only (radius and scale ≥ 1 nm), each a squared hinge; orientations need none
    @test penalty(Ring(), merge(CASES[1][3], (r=0.5, vx=10.0))) ≈ (1 / 0.5^2 - 1)^2
    @test penalty(SphericalCap(), merge(CASES[2][3], (s=0.5, eta=-5.0))) ≈ (1 / 0.5^2 - 1)^2
    # a composite adds its components' penalties; a second ring smaller than the first is not penalized
    @test penalty(DoubleRing(), merge(CASES[3][3], (r1=0.5, r2=0.25))) ≈
        (1 / 0.5^2 - 1)^2 + (1 / 0.25^2 - 1)^2
    @test penalty(DoubleRing(), merge(CASES[3][3], (r2=40.0,))) == 0
    @test penalty(SpCapRing(), merge(CASES[4][3], (s2=0.5,))) ≈ (1 / 0.5^2 - 1)^2
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
        :eta1,
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
        SphericalCap(), U[1], (x=0.0, y=0.0, z=0.0, s=θ.s1, eta=θ.eta1, vx=0.0, vy=0.0)
    )
    ring_alone = model_points(Ring(), U[2], (x=5.0, y=0.0, z=30.0, r=θ.r2, vx=0.3, vy=-0.1))
    @test X ≈ hcat(cap_alone, ring_alone)
    # the world-frame summary of each component
    comps = LocMoFit.components(c, U, θ)
    @test comps[2].position ≈ [5.0, 0.0, 30.0] &&
        comps[2].axis ≈ pose(starts[2], 0.3, -0.1)[:, 3] &&
        comps[2].r == 50.0
    @test comps[1].kappa ≈ tanh(θ.eta1) &&
        comps[1].radius ≈ θ.s1 / tanh(θ.eta1) &&
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
    pf, uf = problem(model, θ, starts, locs, precs; freeze=(:z, :sigma))
    uf[:r] = 45.0
    @test loss(uf, pf) ≈ loss(
        problem(model, merge(θ, (r=45.0,)), starts, locs, precs)[2],
        problem(model, θ, starts, locs, precs)[1],
    )
    @test_throws ArgumentError ParameterMap(θ, (:w,))
    @test_throws ArgumentError ParameterMap(θ, keys(θ))
    # theta, phi set the starting axis; the orientation is frozen through its chart coordinates
    @test_throws ArgumentError problem(model, θ, starts, locs, precs; freeze=(:theta,))
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
    # Enzyme's gradient of the loss (through the rule for the log-likelihood) against central differences, for every model (with the background, the prior and the weights); DoubleRing with its radii tied and its second orientation frozen, and once without a background; SpCapRing also in 2D
    for (_, model, θ, starts, kwargs, D) in (
        (CASES[1]..., (;), 3),
        (CASES[2]..., (;), 3),
        (CASES[3]..., (freeze=(:vx2, :vy2), tie=(:r2 => :r1,)), 3),
        (CASES[4]..., (;), 3),
        (CASES[3]..., (background=false,), 3),
        (CASES[4]..., (;), 2),
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
    init = merge(θ, (x=0.0, z=0.0, r=30.0, sigma=1.0), axes_init(model, starts))
    r = run_locmofit(
        "Ring",
        locs,
        precs;
        init_params=init,
        freeze=(:sigma,),
        dtype=Float64,
        spacing=SPACING,
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
        Ring(),
        locs,
        precs;
        init_params=init,
        freeze=(:sigma,),
        dtype=Float64,
        spacing=SPACING,
    )
    @test r.loss < loss(u0, p0)
    @test r.unit_model.points == unit_points(model, init; spacing=SPACING)   # not changed by the fit
    @test r.model_points ≈ model_points(model, r.unit_model, r.parameters)

    h = run_locmofit(
        "Ring",
        locs,
        precs;
        init_params=init,
        freeze=(:sigma,),
        dtype=Float64,
        spacing=SPACING,
        history=true,
    )
    @test h.parameters == r.parameters
    @test length(h.history.losses) == length(h.history.parameters) > 1
    @test all(q -> q.sigma == 1.0, h.history.parameters)

    # without a background: no signal fraction to fit; a starting axis pointing down works too
    nb = run_locmofit(
        "Ring",
        locs,
        precs;
        init_params=merge(init, (theta=π,)),
        background=false,
        dtype=Float64,
        spacing=SPACING,
        max_iter=5,
    )
    @test !haskey(nb.parameters, :logit_prob) &&
        nb.signal_fraction === nothing &&
        isfinite(nb.loss)

    # freezing works for every model (it failed for SphericalCap and DoubleRing in the Lux version)
    for (name, model, θ, starts) in CASES
        locs, precs = test_data(model, θ, starts)
        f = run_locmofit(
            name,
            locs,
            precs;
            init_params=merge(θ, axes_init(model, starts)),
            freeze=(:sigma,),
            dtype=Float64,
            spacing=SPACING,
            max_iter=5,
        )
        @test f.parameters.sigma == θ.sigma
    end

    # a tie holds throughout, and the unit points are built with it; a composite can be passed as a model, and its shares sum to one
    (_, model, θ, starts) = CASES[3]
    locs, precs = test_data(model, θ, starts)
    t = run_locmofit(
        Composite(Ring(), Ring()),
        locs,
        precs;
        init_params=merge(θ, axes_init(model, starts)),
        tie=(:r2 => :r1,),
        dtype=Float64,
        spacing=SPACING,
        max_iter=20,
    )
    @test t.parameters.r2 == t.parameters.r1
    @test size(t.unit_model[2].points) == size(t.unit_model[1].points)
    @test sum(t.weights) ≈ 1 && length(t.weights) == 2
end

include("fit.jl")
