# Priors on every quantity a fit estimates, and `ParamToFit`, which gives each quantity its starting value and its prior (derivations/derivation.qmd, Priors on every parameter). A prior is a density of a quantity of the structure (a position, a size, a curvature, an axis, the signal fraction, the shares), whatever coordinates the optimizer uses for it, and the loss adds its log density with no Jacobian (@eq-full-posterior).

"""
    Prior

A prior on one quantity of a fit, or on a group of them (an axis, the shares), given in [`ParamToFit`](@ref). Its log density, up to a constant, is added to the log-likelihood (derivations/derivation.qmd, @eq-full-posterior). The densities: [`Flat`](@ref), [`Normal`](@ref), [`LogNormal`](@ref), [`Beta`](@ref), [`Dirichlet`](@ref), [`Watson`](@ref); and the point masses [`Fixed`](@ref) and [`Tied`](@ref), which hold a quantity at its starting value or at another quantity's value.
"""
abstract type Prior end

"""
    Flat()

A constant density: no prior term. It is proper on a bounded quantity (a cap's `kappa` in (−1, 1), an axis on the sphere, the signal fraction, the shares) and improper on a position or a size, which the localizations must then determine.
"""
struct Flat <: Prior end

"""
    Normal(mean, sd)

The normal density of a position (nm), a size (nm; of its magnitude, so that `Normal(0, sd)` on `sigma` is a half-normal) or a cap's `kappa` (restricted to (−1, 1)).
"""
struct Normal{T<:Real} <: Prior
    mean::T
    sd::T
    function Normal(mean::Real, sd::Real)
        sd > 0 || throw(ArgumentError("Normal: sd must be positive, got $sd"))
        T = promote_type(typeof(mean), typeof(sd))
        return new{T}(mean, sd)
    end
end

"""
    LogNormal(mu, sigma)

The log-normal density of a size (nm): its logarithm is normal with mean `mu` and standard deviation `sigma`, so its median is exp(`mu`) and it spreads by a factor exp(`sigma`) (derivations/derivation.qmd, @eq-lognormal). It vanishes at 0, so a size with this prior cannot shrink to a point. The default of a ring's `r` and a cap's `s` is `LogNormal(log(50), 1)`: median 50 nm, 95% of it between 7 and 355 nm.
"""
struct LogNormal{T<:Real} <: Prior
    mu::T
    sigma::T
    function LogNormal(mu::Real, sigma::Real)
        sigma > 0 || throw(ArgumentError("LogNormal: sigma must be positive, got $sigma"))
        T = promote_type(typeof(mu), typeof(sigma))
        return new{T}(mu, sigma)
    end
end

"""
    Beta(a, b)

The Beta density of the signal fraction, with `a, b ≥ 1` (below 1 it grows without bound at 0 or 1). It acts like a + b − 2 extra localizations, a − 1 of them signal (derivations/derivation.qmd, @eq-posterior). The default is `Beta(5, 2)`: mode 0.8, as much as 5 localizations.
"""
struct Beta{T<:Real} <: Prior
    a::T
    b::T
    function Beta(a::Real, b::Real)
        a >= 1 && b >= 1 || throw(
            ArgumentError(
                "Beta: a and b must be at least 1, got ($a, $b): below 1 the density grows without bound at 0 or 1",
            ),
        )
        T = promote_type(typeof(a), typeof(b))
        return new{T}(a, b)
    end
end

"""
    Dirichlet(alpha)

The symmetric Dirichlet density of a composite's shares, with `alpha ≥ 1` (below 1 it grows without bound as a share goes to 0). It acts like alpha − 1 extra signal localizations for each component and keeps every share above 0, so that a component cannot be switched off (derivations/derivation.qmd, @eq-share-prior). The default is `Dirichlet(5)`.
"""
struct Dirichlet{T<:Real} <: Prior
    alpha::T
    function Dirichlet(alpha::Real)
        alpha >= 1 || throw(
            ArgumentError(
                "Dirichlet: alpha must be at least 1, got $alpha: below 1 the density grows without bound as a share goes to 0",
            ),
        )
        return new{typeof(alpha)}(alpha)
    end
end

"""
    Watson(theta, phi, kappa)

The Watson density of an axis on the sphere, ∝ exp(`kappa` (μ·n)²), with μ the unit vector at polar angle `theta` and azimuth `phi` (derivations/derivation.qmd, @eq-watson). It gives n and −n the same density, as the likelihood does. `kappa > 0` keeps the axis near ±μ, `kappa < 0` near the great circle perpendicular to μ (with `theta = 0`, an axis near the xy plane), and `kappa = 0` is uniform. For component k > 1 of a composite, μ is in component 1's frame, as the axis is.
"""
struct Watson{T<:Real} <: Prior
    theta::T
    phi::T
    kappa::T
    function Watson(theta::Real, phi::Real, kappa::Real)
        T = promote_type(typeof(theta), typeof(phi), typeof(kappa))
        return new{T}(theta, phi, kappa)
    end
end

"""
    Fixed()

A point mass at the quantity's starting value: the quantity is not fitted. Without a value it is held at its default.
"""
struct Fixed <: Prior end

"""
    Tied(source)

A point mass at another quantity's value, such as `Tied(:r1)` for `r2`: the quantity takes the source's value throughout, and only the source's prior applies. Both must be of the same kind (positions, sizes or curvatures).
"""
struct Tied <: Prior
    source::Symbol
end
Tied(source::AbstractString) = Tied(Symbol(source))

"""
    ParamToFit(value, prior)
    ParamToFit(value)
    ParamToFit(prior)

What is known about one quantity of a fit before it: its starting `value` (`nothing` to estimate it from the localizations, or else start from its default) and its `prior` (a [`Prior`](@ref); `nothing` for the quantity's default prior). Both are in the quantity's own terms: a position or a size in nm, a cap's `kappa` (κ′), an axis as `(theta, phi)`, the signal fraction, the shares as a tuple. [`run_locmofit`](@ref) lists the quantities and their default priors.
"""
struct ParamToFit{V,P<:Union{Nothing,Prior}}
    value::V
    prior::P
end
ParamToFit(value) = ParamToFit(value, nothing)
ParamToFit(prior::Prior) = ParamToFit(nothing, prior)

# The quantities priors are stated in, each computed from the parameters θ (a NamedTuple) the optimizer varies: their names are type parameters, so that reading them is plain field access in the loss.

abstract type Quantity end

# a position (nm), the parameter n
struct Location{n} <: Quantity end
# a size (nm), the magnitude of the parameter n, which the likelihood is even in
struct Size{n} <: Quantity end
# a cap's curvature κ′ = sign(s)·tanh(η), with η the parameter n and s the parameter s: the cap of scale -s is the cap of scale s with the opposite curvature
struct Curvature{n,s} <: Quantity end
# the axis of component k, from its chart coordinates vx, vy: as m = normalize((vx, vy, 1)), the axis in the chart's frame
struct Axis{k,vx,vy} <: Quantity end
# the signal fraction π = logistic(logit_prob)
struct SignalFraction <: Quantity end
# the shares softmax(0, ω₂, …), with the parameters ω₂, … named `names`
struct Shares{names} <: Quantity end

# the log of a fraction and of its complement, which the Beta density needs, formed without the fraction (derivations/derivation.qmd, @eq-logfractions)
struct LogFraction{T}
    log::T
    log1m::T
end

quantity(::Location{n}, θ) where {n} = getfield(θ, n)
quantity(::Size{n}, θ) where {n} = abs(getfield(θ, n))
quantity(::Curvature{n,s}, θ) where {n,s} = sign(getfield(θ, s)) * tanh(getfield(θ, n))
function quantity(::Axis{k,vx,vy}, θ) where {k,vx,vy}
    a, b = getfield(θ, vx), getfield(θ, vy)
    r = inv(sqrt(1 + a^2 + b^2))
    return (a * r, b * r, r)
end
function quantity(::SignalFraction, θ)
    return LogFraction(-softplus(-θ.logit_prob), -softplus(θ.logit_prob))
end
function quantity(::Shares{names}, θ) where {names}
    lw = (zero(θ.sigma), ntuple(i -> getfield(θ, names[i]), Val(length(names)))...)
    return lw .- logsumexp(lw)
end

# The Watson density in the chart's coordinates: μ·n = (R₀ᵀμ)·m for the axis n = R₀m, so it needs only m and R₀ᵀμ, which is fixed for the whole fit (derivations/derivation.qmd, Axes)
struct ChartWatson{T} <: Prior
    mu::NTuple{3,T}
    kappa::T
end

# the log densities, up to a constant
log_density(p::Normal, q) = -((q - p.mean) / p.sd)^2 / 2
function log_density(p::LogNormal, q)
    l = log(q)
    return -l - ((l - p.mu) / p.sigma)^2 / 2
end
log_density(p::Beta, f::LogFraction) = (p.a - 1) * f.log + (p.b - 1) * f.log1m
log_density(p::Dirichlet, logw::Tuple) = (p.alpha - 1) * sum(logw)
function log_density(p::ChartWatson, m)
    return p.kappa * (p.mu[1] * m[1] + p.mu[2] * m[2] + p.mu[3] * m[3])^2
end

"""
    PriorTerm(quantity, prior)

One term of the log prior in [`loss`](@ref): the log density of `prior` at the `quantity` computed from the parameters.
"""
struct PriorTerm{Q<:Quantity,P<:Prior}
    quantity::Q
    prior::P
end

"""
    log_prior(terms, θ)

The sum of the prior terms (a tuple of [`PriorTerm`](@ref)) at the parameters θ; 0 for none.
"""
log_prior(::Tuple{}, θ) = zero(θ.sigma)
log_prior(terms::Tuple, θ) = log_prior(first(terms), θ) + log_prior(Base.tail(terms), θ)
log_prior(t::PriorTerm, θ) = log_density(t.prior, quantity(t.quantity, θ))

# which priors each kind of quantity takes (Flat and Fixed fit every kind)
supports(::Quantity, ::Union{Flat,Fixed}) = true
supports(::Union{Location,Size,Curvature}, ::Union{Normal,Tied}) = true
supports(::Size, ::LogNormal) = true
supports(::Axis, ::Watson) = true
supports(::SignalFraction, ::Beta) = true
supports(::Shares, ::Dirichlet) = true
supports(::Quantity, ::Prior) = false

describe(::Location) = "position"
describe(::Size) = "size"
describe(::Curvature) = "curvature"
describe(::Axis) = "axis"
describe(::SignalFraction) = "signal fraction"
describe(::Shares) = "set of shares"

# the parameters a quantity is computed from
parameter_names(::Location{n}) where {n} = (n,)
parameter_names(::Size{n}) where {n} = (n,)
parameter_names(::Curvature{n}) where {n} = (n,)
parameter_names(::Axis{k,vx,vy}) where {k,vx,vy} = (vx, vy)
parameter_names(::SignalFraction) = (:logit_prob,)
parameter_names(::Shares{names}) where {names} = names

# A quantity's value as starting values of the parameters; `axes` are the names that set each component's starting axis (orientation_names)
start_values(q::Union{Location,Size}, v::Real, axes) = (only(parameter_names(q)) => v,)
function start_values(q::Curvature, v::Real, axes)
    abs(v) < 1 || throw(ArgumentError("kappa must lie in (-1, 1), got $v"))
    return (only(parameter_names(q)) => atanh(v),)
end
function start_values(::Axis{k}, v, axes) where {k}
    length(v) == 2 ||
        throw(ArgumentError("an axis is given as (theta, phi), got $(repr(v))"))
    return (axes[k][1] => v[1], axes[k][2] => v[2])
end
function start_values(::SignalFraction, v::Real, axes)
    0 < v < 1 || throw(ArgumentError("the signal fraction must lie in (0, 1), got $v"))
    return (:logit_prob => log(v) - log1p(-v),)
end
function start_values(::Shares{names}, v, axes) where {names}
    length(v) == length(names) + 1 && all(>(0), v) || throw(
        ArgumentError(
            "the shares are given as $(length(names) + 1) positive numbers, got $(repr(v))",
        ),
    )
    return Tuple(n => log(v[i + 1]) - log(v[1]) for (i, n) in enumerate(names))
end
function start_values(q::Quantity, v, axes)
    return throw(ArgumentError("a $(describe(q)) is not given as $(repr(v))"))
end

const SIZE_PRIOR = LogNormal(log(50.0), 1.0)
const SIGNAL_PRIOR = Beta(5.0, 2.0)
const SHARE_PRIOR = Dirichlet(5.0)

# The quantities of a simple model as component `index` of a fit, whose parameter names end in `suffix` ("" for a model fitted alone), each with its default prior: its position, its shape (shape_quantities) and its axis.
function quantities(m::SimpleModel, suffix, index)
    return (
        (
            Symbol(n, suffix) => (Location{Symbol(n, suffix)}(), Flat()) for
            n in (:x, :y, :z)
        )...,
        shape_quantities(m, suffix)...,
        Symbol(:axis, suffix) =>
            (Axis{index,Symbol(:vx, suffix),Symbol(:vy, suffix)}(), Flat()),
    )
end

"""
    quantities(model, background)

The quantities of a fit of `model`, each as `name => (quantity, default prior)`: those of every component (`quantities(component, suffix, index)`), the blur `sigma`, a composite's `shares`, and with a background the `signal` fraction.
"""
function quantities(m::SimpleModel, background::Bool)
    return (quantities(m, "", 1)..., common_quantities(m, background)...)
end
function common_quantities(m::Model, background)
    return (
        :sigma => (Size{:sigma}(), Flat()),
        share_quantities(m)...,
        (background ? (:signal => (SignalFraction(), SIGNAL_PRIOR),) : ())...,
    )
end
share_quantities(::SimpleModel) = ()

"""
    specification(model, background, params, axes)

What `params` (a `NamedTuple` of [`ParamToFit`](@ref)) asks of a fit of `model`, in the optimizer's parameters: the starting values (`init`, with each starting axis under its `axes` names), the parameters held (`frozen`), the ties (`tie`, `target => source`), and the priors as `quantity => prior` (`priors`; flat ones left out). Throws an `ArgumentError` for an unknown name, a prior that does not fit its quantity, or a value outside its quantity's range.
"""
function specification(model::Model, background::Bool, params::NamedTuple, axes)
    table = quantities(model, background)
    names = map(first, table)
    unknown = Tuple(k for k in keys(params) if !(k in names))
    isempty(unknown) || throw(
        ArgumentError("unknown quantities $unknown; the quantities of this fit are $names"),
    )
    kinds = Dict(table)
    init = Pair{Symbol,Float64}[]
    frozen, tie, priors = Symbol[], Pair{Symbol,Symbol}[], Pair{Quantity,Prior}[]
    for (name, (q, default)) in table
        spec = get(params, name, ParamToFit(nothing, nothing))
        spec isa ParamToFit ||
            throw(ArgumentError("params.$name must be a ParamToFit, got $(repr(spec))"))
        prior = spec.prior === nothing ? default : spec.prior
        supports(q, prior) || throw(
            ArgumentError(
                "$name is a $(describe(q)), which does not take a $(nameof(typeof(prior))) prior",
            ),
        )
        spec.value === nothing || append!(init, start_values(q, spec.value, axes))
        if prior isa Fixed
            append!(frozen, parameter_names(q))
        elseif prior isa Tied
            haskey(kinds, prior.source) && prior.source != name || throw(
                ArgumentError(
                    "$name is tied to $(prior.source), which is not another quantity of this fit",
                ),
            )
            source = first(kinds[prior.source])
            nameof(typeof(source)) == nameof(typeof(q)) || throw(
                ArgumentError(
                    "$name, a $(describe(q)), is tied to $(prior.source), a $(describe(source))",
                ),
            )
            push!(tie, only(parameter_names(q)) => only(parameter_names(source)))
        elseif !(prior isa Flat)
            push!(priors, q => prior)
        end
    end
    return (init=NamedTuple(init), frozen=Tuple(frozen), tie=Tuple(tie), priors=priors)
end

"""
    prior_terms(priors, unit_model, T)

The priors of [`specification`](@ref) as the tuple of [`PriorTerm`](@ref)s the loss evaluates, in the floating-point type `T`. A Watson prior on an axis becomes a density in the chart's coordinates, which needs the chart's frame from the unit model.
"""
function prior_terms(priors, unit, T)
    return Tuple(prior_term(q, p, unit, T) for (q, p) in priors)
end
prior_term(q, p::Normal, unit, T) = PriorTerm(q, Normal(T(p.mean), T(p.sd)))
prior_term(q, p::LogNormal, unit, T) = PriorTerm(q, LogNormal(T(p.mu), T(p.sigma)))
prior_term(q, p::Beta, unit, T) = PriorTerm(q, Beta(T(p.a), T(p.b)))
prior_term(q, p::Dirichlet, unit, T) = PriorTerm(q, Dirichlet(T(p.alpha)))
function prior_term(q::Axis{k}, p::Watson, unit, T) where {k}
    R₀ = unit isa Tuple ? unit[k].chart : unit.chart
    μ = R₀' * axis_vector(p.theta, p.phi)
    return PriorTerm(q, ChartWatson(Tuple(T.(μ)), T(p.kappa)))
end
