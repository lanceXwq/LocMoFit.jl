# The likelihood of derivations/derivation.qmd: each localization's density is a mixture of the structure (the mean of Gaussians over each component's model points, the components weighted) and a uniform background.

"""
    loss(u, p)

The objective of a fit: minus the log posterior of the parameters, per localization, plus the model's penalty. `u` holds the free parameters; `p` (from [`fit_problem`](@ref)) holds the model, the map to all parameters, the unit model (unit points and chart frames) and the rows of each component's points (`ranges`), the localizations and their precisions (`D×N`), and the background: whether there is one, the log of the field of view's volume, and the Beta prior on the signal fraction.
"""
function loss(u, p)
    θ = full_parameters(u, p.parameter_map)
    points = model_points(p.model, p.unit_model, θ)
    lw = log_weights(p.model, θ)
    logw = lw .- logsumexp(lw)
    N = size(p.locs, 2)
    if !p.background
        ℓ = log_likelihood(
            p.locs,
            p.loc_prec,
            points,
            p.ranges,
            θ.sigma,
            logw,
            zero(θ.sigma),
            oftype(θ.sigma, -Inf),
        )
        return -ℓ / N + penalty(p.model, θ)
    end
    logπ, log1mπ = -softplus(-θ.logit_prob), -softplus(θ.logit_prob)
    ℓ = log_likelihood(
        p.locs, p.loc_prec, points, p.ranges, θ.sigma, logw, logπ, log1mπ - p.log_volume
    )
    prior = if p.signal_prior === nothing
        zero(logπ)
    else
        (p.signal_prior[1] - 1) * logπ + (p.signal_prior[2] - 1) * log1mπ
    end
    return -(ℓ + prior) / N + penalty(p.model, θ)
end

"""
    log_likelihood(locs, loc_prec, points, ranges, σ, logw, a, b)

Σₙ log(eᵃ pₙ + eᵇ) over the localizations (the columns of the `D×N` `locs`, with precisions `loc_prec`), where pₙ is the structure's density at localization n: for each component k, whose model points are the columns `ranges[k]` of `points` (of which the first `D` rows are used), the mean of the Gaussians around its points with per-axis variances `loc_prec² + σ²`; the components weighted by `exp.(logw)`. In a fit, `a` is the log signal fraction and `b` the log of the background's density times its fraction; `b = -Inf` means no background.

The squared distances are formed from differences (not expanded), so the result keeps its precision far from the origin, also in `Float32`. For each localization the terms of its model points are kept, so that [`likelihood_pass`](@ref) computes the gradient from them without evaluating them again; Enzyme uses that gradient through the rule in `rules.jl`.
"""
function log_likelihood(locs, loc_prec, points, ranges, σ, logw, a, b)
    return first(likelihood_pass(locs, loc_prec, points, ranges, σ, logw, a, b, Val(false)))
end

"""
    likelihood_pass(locs, loc_prec, points, ranges, σ, logw, a, b, Val(gradient)) -> (ℓ, ∇ℓ)

[`log_likelihood`](@ref) and, with `Val(true)`, its gradient with respect to `points` (`3×M`), `σ`, `logw`, `a` and `b`, as a `NamedTuple` (`nothing` with `Val(false)`). For each localization, the terms of every component are kept in a buffer, one per point, and once the localization's density is known one more pass over them gives the derivatives with respect to the points and the variances (derivations/derivation.qmd).
"""
function likelihood_pass(
    locs::AbstractMatrix, loc_prec, points, ranges, σ, logw, a, b, gradient::Val
)
    return if size(locs, 1) == 3
        likelihood_pass(Val(3), locs, loc_prec, points, ranges, σ, logw, a, b, gradient)
    else
        likelihood_pass(Val(2), locs, loc_prec, points, ranges, σ, logw, a, b, gradient)
    end
end

function likelihood_pass(
    ::Val{D},
    locs::AbstractMatrix{T},
    loc_prec::AbstractMatrix{T},
    points::AbstractMatrix{T},
    ranges::NTuple{K,UnitRange{Int}},
    σ::T,
    logw::NTuple{K,T},
    a::T,
    b::T,
    ::Val{G},
) where {D,T,K,G}
    M = size(points, 2)
    Y = ntuple(d -> points[d, :], Val(D))                  # each coordinate of the points, contiguous
    dY = ntuple(_ -> zeros(T, G ? M : 0), Val(D))
    E = Vector{T}(undef, M)                                # one localization's terms, one per point
    lognorm = map((lw, r) -> lw - log(T(length(r))), logw, ranges)                      # log(wₖ/Mₖ)
    ℓ = dσ = da = db = zero(T)
    dlogw = ntuple(_ -> zero(T), Val(K))
    for n in axes(locs, 2)
        x = ntuple(d -> locs[d, n], Val(D))
        v = ntuple(d -> loc_prec[d, n]^2 + σ^2, Val(D))
        h = map(v -> inv(2v), v)
        c = -sum(map(v -> log(T(2π) * v), v)) / 2        # the Gaussians' normalization
        parts = map(r -> component_terms!(E, Y, x, h, r), ranges)
        t = map((ln, part) -> ln + part.lse, lognorm, parts)
        S = logsumexp(t)
        L = logaddexp(a + c + S, b)
        ℓ += L
        if G
            ρ = exp(a + c + S - L)                         # the probability that the localization is signal
            da += ρ
            db += 1 - ρ
            g = map(t -> ρ * exp(t - S), t)                # ∂L/∂ each component's lse
            dlogw = dlogw .+ g
            dh = ntuple(_ -> zero(T), Val(D))
            for k in 1:K
                dh =
                    dh .+
                    component_gradient!(dY, E, Y, x, h, ranges[k], g[k] / parts[k].sum)
            end
            # v = s² + σ² enters c (∂c/∂v = -h) and h (∂h/∂v = -2h²)
            dσ += 2σ * sum(map((h, dh) -> -ρ * h - 2h^2 * dh, h, dh))
        end
    end
    G || return ℓ, nothing
    dpoints = zeros(T, size(points))
    for d in 1:D
        dpoints[d, :] .= dY[d]
    end
    return ℓ, (points=dpoints, σ=dσ, logw=dlogw, a=da, b=db)
end

# Σ_d h_d (y_dm - x_d)², the exponent of point m's Gaussian. Here and in the loops below, `@inbounds` is written inside every closure: it does not reach into them from the loop, and a bounds check stops the loop from vectorizing.
@inline sqdist(Y::NTuple{D}, x, h, m) where {D} =
    sum(ntuple(d -> @inbounds(h[d] * (Y[d][m] - x[d])^2), Val(D)))

# lse = log Σₘ exp(-qₘ) over the points `r`, with qₘ = sqdist; leaves the terms exp(q_min - qₘ) in E and returns their sum. Two passes (the smallest qₘ, then the terms), which, unlike a running maximum, vectorize.
@inline function component_terms!(E, Y, x, h, r)
    qmin = typemax(eltype(E))
    @inbounds @simd for m in r
        q = sqdist(Y, x, h, m)
        E[m] = q
        qmin = @fastmath min(qmin, q)
    end
    acc = zero(eltype(E))
    @inbounds @simd for m in r
        e = exp_nonpositive(qmin - E[m])
        E[m] = e
        acc += e
    end
    return (lse=log(acc) - qmin, sum=acc)
end

# Adds g·∂lse/∂y_dm to dY and returns g·∂lse/∂h_d, for c = g/acc (g = ∂L/∂lse, acc the sum of the terms in E): with the weights wₘ = E[m]/acc, ∂lse/∂y_dm = -2h_d wₘ (y_dm - x_d) and ∂lse/∂h_d = -Σₘ wₘ (y_dm - x_d)².
@inline function component_gradient!(dY::NTuple{D}, E, Y, x, h, r, c) where {D}
    f = map(h -> -2h * c, h)
    s = ntuple(_ -> zero(c), Val(D))
    @inbounds @simd for m in r
        w = E[m]
        δ = ntuple(d -> @inbounds(Y[d][m] - x[d]), Val(D))
        ntuple(d -> @inbounds(dY[d][m] += f[d] * w * δ[d]), Val(D))
        s = s .+ map(δ -> w * δ^2, δ)
    end
    return map(s -> -c * s, s)
end

# the Taylor coefficients 1/k! of exp, as many as each precision needs
const EXP_TAYLOR_64 = ntuple(k -> 1 / factorial(k - 1), 14)
const EXP_TAYLOR_32 = ntuple(k -> 1.0f0 / factorial(k - 1), 8)

"""
    exp_nonpositive(x)

`exp(x)` for `x ≤ 0`, within one unit in the last place (as `Base.exp`), and 0 where it would be subnormal. Unlike `Base.exp`, which looks up a table, it is plain arithmetic, so that a loop over it vectorizes: x = k·log 2 + r with |r| ≤ log(2)/2, the Taylor polynomial of exp(r), and 2ᵏ from the exponent bits. log 2 is split in two (Cody and Waite) so that k·log 2 is exact enough without a fused multiply-add.
"""
@inline function exp_nonpositive(x::Float64)
    k = round(x * 1.4426950408889634)
    r = muladd(k, -1.9082149292705877e-10, muladd(k, -0.6931471803691238, x))
    # inlined by hand: at this length evalpoly is no longer inlined by itself, and the call would stop vectorization
    p = @inline evalpoly(r, EXP_TAYLOR_64)
    s = reinterpret(Float64, (unsafe_trunc(Int64, k) + 1023) << 52)
    return ifelse(x < -708.0, 0.0, p * s)
end
@inline function exp_nonpositive(x::Float32)
    k = round(x * 1.442695f0)
    r = muladd(k, 2.1219444f-4, muladd(k, -0.693359375f0, x))
    p = evalpoly(r, EXP_TAYLOR_32)
    s = reinterpret(Float32, (unsafe_trunc(Int32, k) + Int32(127)) << 23)
    return ifelse(x < -87.0f0, 0.0f0, p * s)
end

function logsumexp(x::Tuple)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end
logaddexp(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))      # log(1 + eˣ)

sq_hinge(x) = max(0, x)^2
