# Starting values estimated from the localizations (derivations/derivation.qmd, Starting values from the data)

const START_COARSE = 100            # candidate axes spread over the hemisphere
const START_FINE = 64               # candidate axes around the best coarse one,
const START_CONE = deg2rad(25)      # within this angle of it
const START_MAX_KAPPA = 0.95        # |κ′| of a starting cap, which keeps η = atanh κ′ finite

"""
    estimate_start(model, locs, loc_precisions, axis) -> NamedTuple

Starting values for `model` estimated from one site's localizations (`D×N` positions and precisions, nm), as `init_params` takes them: in 3D the position, the starting axis (`theta`, `phi`) and the shape of a `Ring` or a `SphericalCap`, and for a composite of two of them sharing an axis also the second component's shape and distance along the axis (`z2`) and the shares (`omega2`); in 2D only the position. `axis`, a unit vector or `nothing`, is a starting axis given by the user, along which the rest is estimated; with `nothing` the axis is estimated too. Any other model gets only the position of its component 1, at the weighted center of the localizations, so that its fit starts at the site rather than at the origin; the rest starts from its defaults. Values that come out undefined (too few localizations) are left out.
"""
function estimate_start(m::Model, locs, loc_precisions, axis)
    return position_start(m, locs, loc_precisions)
end

# the position of component 1 at the weighted center of the localizations
function position_start(m::Model, locs, loc_precisions)
    f = site_frame(locs, loc_precisions)
    f === nothing && return NamedTuple()
    return finite_values(NamedTuple{position_names(m, size(locs, 1))}(Tuple(f.center)))
end
position_names(::SimpleModel, D) = D == 2 ? (:x, :y) : (:x, :y, :z)
position_names(::Composite, D) = D == 2 ? (:x1, :y1) : (:x1, :y1, :z1)

function estimate_start(m::Union{Ring,SphericalCap}, locs, loc_precisions, axis)
    f = site_frame(locs, loc_precisions)
    f === nothing && return NamedTuple()
    size(locs, 1) == 2 && return finite_values((x=f.center[1], y=f.center[2]))
    n = axis === nothing ? principal_axis(f) : axis
    return finite_values(simple_start(m, f, n, axis === nothing))
end

function estimate_start(
    c::Composite{<:Tuple{SimpleModel,SimpleModel}}, locs, loc_precisions, axis
)
    parts = c.components
    all(p -> p isa Union{Ring,SphericalCap}, parts) ||
        return position_start(c, locs, loc_precisions)
    f = site_frame(locs, loc_precisions)
    f === nothing && return NamedTuple()
    size(locs, 1) == 2 && return finite_values((x1=f.center[1], y1=f.center[2]))
    s = AxisScores(f)
    n, first_low =
        axis === nothing ? search_axis(parts, s) : (axis, score!(s, parts, axis)[2])
    score!(s, parts, n)                    # the groups along the chosen axis
    low, high = view(s.order, 1:s.nlow), view(s.order, (s.nlow + 1):length(s.order))
    # a cap's parabola fits a ring's group as well (all at one height and distance), so a ring and a cap are told apart by the spread of the distances from the axis: least, relative to their median, in the ring's group
    if count(p -> p isa Ring, parts) == 1
        ring_low = relative_spread(s.ρ[low]) < relative_spread(s.ρ[high])
        first_low = (parts[1] isa Ring) == ring_low
    end
    g1, g2 = first_low ? (low, high) : (high, low)
    # an estimated axis points from component 1 to component 2, so that z2 > 0 as in the presets
    if axis === nothing && median_of(s.w[g2]) < median_of(s.w[g1])
        n = -n
        s.w .= -s.w
    end
    t1, shape1 = part_start(parts[1], f, s, g1, n)
    t2, shape2 = part_start(parts[2], f, s, g2, n)
    θ1, ϕ1 = axis_angles(n)
    start = (
        x1=t1[1],
        y1=t1[2],
        z1=t1[3],
        theta1=θ1,
        phi1=ϕ1,
        suffixed(shape1, 1)...,
        z2=dot(t2 - t1, n),
        suffixed(shape2, 2)...,
        omega2=log(length(g2) / length(g1)),
    )
    return finite_values(start)
end

# the names estimate_start gives a model in D dimensions, so that fit_problem can skip it when they are all given
start_names(m::Model, D) = position_names(m, D)
function start_names(m::Union{Ring,SphericalCap}, D)
    D == 2 && return (:x, :y)
    return (:x, :y, :z, keys(shape_parameters(m, Float64))..., :theta, :phi)
end
function start_names(c::Composite{<:Tuple{SimpleModel,SimpleModel}}, D)
    all(p -> p isa Union{Ring,SphericalCap}, c.components) || return position_names(c, D)
    D == 2 && return (:x1, :y1)
    s1, s2 = (keys(suffixed(shape_parameters(c.components[k], Float64), k)) for k in 1:2)
    return (:x1, :y1, :z1, :theta1, :phi1, s1..., :z2, s2..., :omega2)
end

function suffixed(shape::NamedTuple, k)
    return NamedTuple{map(n -> Symbol(n, k), keys(shape))}(values(shape))
end
finite_values(v::NamedTuple) = NamedTuple(k => x for (k, x) in pairs(v) if isfinite(x))

axis_vector(θ, ϕ) = [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
axis_angles(n) = (acos(clamp(n[3], -1, 1)), atan(n[2], n[1]))

# the standard deviation over the median
function relative_spread(v)
    μ = sum(v) / length(v)
    return sqrt(sum(x -> (x - μ)^2, v) / length(v)) / median_of(v)
end

function median_of(v)
    s = sort(v)
    k = length(s)
    return isodd(k) ? s[(k + 1) ÷ 2] : (s[k ÷ 2] + s[k ÷ 2 + 1]) / 2
end

"""
    site_frame(locs, loc_precisions)

The localizations within twice the median distance from their coordinate-wise median, as `(rel, var, c, center)`: their positions relative to their weighted mean `center`, their precision variances, and their weights `c`, the inverse of each one's summed precision variance, normalized to sum to 1. `nothing` with fewer than 4 left.
"""
function site_frame(locs, loc_precisions)
    X = Matrix{Float64}(locs)
    V = Matrix{Float64}(loc_precisions) .^ 2
    med = [median_of(X[d, :]) for d in axes(X, 1)]
    dist = vec(sqrt.(sum(abs2, X .- med; dims=1)))
    keep = dist .< 2 * median_of(dist)
    count(keep) < 4 && return nothing
    X, V = X[:, keep], V[:, keep]
    c = 1 ./ vec(sum(V; dims=1))
    c ./= sum(c)
    center = X * c
    return (rel=X .- center, var=V, c=c, center=center)
end

# the eigenvector of the weighted covariance, less the mean precision variance, with the smallest eigenvalue: the axis of a ring or a cap
function principal_axis(f)
    C = (f.rel .* f.c') * f.rel' - Diagonal(f.var * f.c)
    return eigen(Symmetric(C)).vectors[:, 1]
end

# each localization's height along n and its distance from the axis through the center
function heights(rel, n)
    w = vec(n' * rel)
    ρ = sqrt.(max.(vec(sum(abs2, rel; dims=1)) .- w .^ 2, 0))
    return w, ρ
end

function simple_start(::Ring, f, n, free_axis)
    free_axis && n[3] < 0 && (n = -n)
    _, ρ = heights(f.rel, n)
    θ, ϕ = axis_angles(n)
    return (x=f.center[1], y=f.center[2], z=f.center[3], r=median_of(ρ), theta=θ, phi=ϕ)
end

function simple_start(::SphericalCap, f, n, free_axis)
    w, ρ = heights(f.rel, n)
    a, b = parabola(w, ρ .^ 2)
    # an estimated axis is turned so that the cap bends toward −n (κ′ ≥ 0); turned around, a cap is the same with −κ′
    free_axis && b > 0 && ((n, a, b) = (-n, -a, -b))
    apex = f.center + a * n
    s, η = cap_shape(f.rel .- a * n, f.var, f.c, b)
    θ, ϕ = axis_angles(n)
    return (x=apex[1], y=apex[2], z=apex[3], s=s, eta=η, theta=θ, phi=ϕ)
end

# the least-squares fit w ≈ a + b·x, as (a, b)
function parabola(w, x)
    k = length(w)
    Sx, Sy, Sxx, Sxy = sum(x), sum(w), sum(abs2, x), dot(x, w)
    b = (k * Sxy - Sx * Sy) / (k * Sxx - Sx^2)
    return (Sy - b * Sx) / k, b
end

# a cap's size from E|x − apex|² = 2s² + Σ_d s_d² (d: the localizations relative to the apex), averaged with the weights c, and its curvature −2b
function cap_shape(d, var, c, b)
    m = sum(i -> c[i] * (sum(abs2, view(d, :, i)) - sum(view(var, :, i))), eachindex(c))
    s = sqrt(max(m / (2 * sum(c)), 1))
    return s, atanh(clamp(-2b * s, -START_MAX_KAPPA, START_MAX_KAPPA))
end

# the reference point (a ring's center, a cap's apex) and the shape of a component from its group G, along the axis n
function part_start(::Ring, f, s, G, n)
    return f.center + median_of(s.w[G]) * n, (r=median_of(s.ρ[G]),)
end
function part_start(::SphericalCap, f, s, G, n)
    a, b = parabola(s.w[G], s.ρ2[G])
    sc, η = cap_shape(f.rel[:, G] .- a * n, f.var[:, G], f.c[G], b)
    return f.center + a * n, (s=sc, eta=η)
end

"""
    AxisScores(f)

The buffers the direction search scores candidate axes with, for the localizations of a `site_frame`: their heights `w` along the candidate, distances `ρ` from it and `ρ2` = ρ², the expected squared precision errors along it (`along`) and across it (`across`, per direction), their `order` along it, and the size `nlow` of the lower group.
"""
mutable struct AxisScores{F}
    const f::F
    const w::Vector{Float64}
    const ρ::Vector{Float64}
    const ρ2::Vector{Float64}
    const along::Vector{Float64}
    const across::Vector{Float64}
    const order::Vector{Int}
    nlow::Int
end
function AxisScores(f)
    N = length(f.c)
    return AxisScores(f, zeros(N), zeros(N), zeros(N), zeros(N), zeros(N), collect(1:N), 0)
end

"""
    score!(s, parts, n) -> (score, first_low)

The score of the candidate axis `n` for a composite of two `parts`: the localizations split in two along `n` where the variance between the groups is largest, each group scored as one part (`part_score`), with the parts assigned to the groups in the order that scores lower; `first_low` tells whether the first part is in the lower group.
"""
function score!(s::AxisScores, parts, n)
    rel, var = s.f.rel, s.f.var
    N = length(s.w)
    @inbounds for i in 1:N
        w = n[1] * rel[1, i] + n[2] * rel[2, i] + n[3] * rel[3, i]
        ρ2 = max(rel[1, i]^2 + rel[2, i]^2 + rel[3, i]^2 - w^2, 0.0)
        along = var[1, i] * n[1]^2 + var[2, i] * n[2]^2 + var[3, i] * n[3]^2
        s.w[i], s.ρ2[i], s.ρ[i] = w, ρ2, sqrt(ρ2)
        s.along[i], s.across[i] = along, (var[1, i] + var[2, i] + var[3, i] - along) / 2
    end
    sortperm!(s.order, s.w)
    s.nlow = largest_between_variance(s.w, s.order)
    low, high = view(s.order, 1:s.nlow), view(s.order, (s.nlow + 1):N)
    a = part_score(parts[1], s, low) + part_score(parts[2], s, high)
    b = part_score(parts[1], s, high) + part_score(parts[2], s, low)
    return a <= b ? (a, true) : (b, false)
end

# the number k of lowest values (w in the given order) whose split from the rest maximizes the between-group variance k(N − k)(mean₁ − mean₂)²
function largest_between_variance(w, order)
    N = length(order)
    total = sum(w)
    best, kbest, c = -Inf, 1, 0.0
    for k in 1:(N - 1)
        c += w[order[k]]
        between = k * (N - k) * (c / k - (total - c) / (N - k))^2
        between > best && ((best, kbest) = (between, k))
    end
    return kbest
end

# a ring's group: the weighted squared distance to the best circle around the axis, (ρ − ρ̄)² + (w − w̄)², less the precision errors across and along the axis
function part_score(::Ring, s::AxisScores, G)
    c = s.f.c
    S = sum(i -> c[i], G)
    ρ̄ = sum(i -> c[i] * s.ρ[i], G) / S
    w̄ = sum(i -> c[i] * s.w[i], G) / S
    return sum(i -> c[i] * ((s.ρ[i] - ρ̄)^2 + (s.w[i] - w̄)^2 - s.across[i] - s.along[i]), G)
end

# a cap's group: the weighted least-squares residual of w ≈ a + b·ρ², less the precision error along the axis
function part_score(::SphericalCap, s::AxisScores, G)
    length(G) < 3 && return Inf
    c = s.f.c
    S = sum(i -> c[i], G)
    Sx = sum(i -> c[i] * s.ρ2[i], G)
    Sy = sum(i -> c[i] * s.w[i], G)
    Sxx = sum(i -> c[i] * s.ρ2[i]^2, G)
    Sxy = sum(i -> c[i] * s.ρ2[i] * s.w[i], G)
    det = S * Sxx - Sx^2
    det > eps() * S * Sxx || return Inf
    b = (S * Sxy - Sx * Sy) / det
    a = (Sy - b * Sx) / S
    return sum(i -> c[i] * ((s.w[i] - a - b * s.ρ2[i])^2 - s.along[i]), G)
end

# the axis with the lowest score among directions spread over the hemisphere, then among the best one and directions around it, as (axis, first_low)
function search_axis(parts, s::AxisScores)
    n, _ = best_axis(parts, s, hemisphere(START_COARSE))
    return best_axis(parts, s, hcat(n, cone(n, START_CONE, START_FINE)))
end
function best_axis(parts, s::AxisScores, candidates)
    best, j, first_low = Inf, 1, true
    for k in axes(candidates, 2)
        v, fl = score!(s, parts, view(candidates, :, k))
        v < best && ((best, j, first_low) = (v, k, fl))
    end
    return candidates[:, j], first_low
end

# k directions spread evenly over the hemisphere z > 0 (a Fibonacci lattice); the score does not change with the sign of the axis
function hemisphere(k)
    X = Matrix{Float64}(undef, 3, k)
    for i in 0:(k - 1)
        z = 1 - (i + 0.5) / k
        r = sqrt(1 - z^2)
        X[:, i + 1] .= (r * cos(i * GOLDEN_ANGLE), r * sin(i * GOLDEN_ANGLE), z)
    end
    return X
end

# k directions spread evenly within the angle α of the unit vector n
function cone(n, α, k)
    e1 = normalize(cross(n, abs(n[1]) < 0.9 ? [1.0, 0, 0] : [0, 1.0, 0]))
    e2 = cross(n, e1)
    X = Matrix{Float64}(undef, 3, k)
    for i in 0:(k - 1)
        z = 1 - (1 - cos(α)) * (i + 0.5) / k
        r = sqrt(1 - z^2)
        X[:, i + 1] .=
            r * cos(i * GOLDEN_ANGLE) .* e1 .+ r * sin(i * GOLDEN_ANGLE) .* e2 .+ z .* n
    end
    return X
end
