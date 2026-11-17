# Priors on every quantity a fit estimates, and `ParamToFit`, which gives each quantity its starting value and its prior (derivations/derivation.qmd, Priors on every parameter). A prior is a density of a quantity of the structure (a position, a size, a curvature, an axis, the fractions of the mixture), whatever coordinates the optimizer uses for it, and the loss adds its log density with no Jacobian (@eq-full-posterior).

"""
    Prior

A prior on one quantity of a fit, or on a group of them (an axis, the fractions), given in [`ParamToFit`](@ref). Its log density, up to a constant, is added to the log-likelihood (derivations/derivation.qmd, @eq-full-posterior). The densities: [`Flat`](@ref), [`Normal`](@ref), [`LogNormal`](@ref), [`Dirichlet`](@ref), [`Watson`](@ref), [`Bingham`](@ref); and the point masses [`Fixed`](@ref) and [`Tied`](@ref), which hold a quantity at its starting value or at another quantity's value.
"""
abstract type Prior end

"""
    Flat()

A constant density: no prior term. It is proper on a bounded quantity (a cap's `kappa` in [−1, 1], an axis on the sphere, the fractions) and improper on a position or a size, which the localizations must then determine.
"""
struct Flat <: Prior end

"""
    Normal(mean, sd)

The normal density of a position (nm), a size (nm; of its magnitude, so that `Normal(0, sd)` on `sigma` is a half-normal) or a cap's `kappa` (restricted to [−1, 1]).
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

The log-normal density of a size (nm): its logarithm is normal with mean `mu` and standard deviation `sigma`, so its median is exp(`mu`) and it spreads by a factor exp(`sigma`) (derivations/derivation.qmd, @eq-lognormal). It vanishes at 0, so a size with this prior cannot shrink to a point. Its most probable value, which a fit's maximum favors, is exp(`mu` − `sigma`²). The default of a ring's `r` and a cap's `s` is `LogNormal(log(50) + 1, 1)`: most probable at 50 nm, median 136 nm, 95% of it between 19 and 965 nm.
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
    Dirichlet(alpha)
    Dirichlet(alpha0, alpha1, …, alphaK)

The Dirichlet density of the fractions of a fit's mixture, ∝ ∏ₖ fₖ^(alphaₖ − 1) (derivations/derivation.qmd, @eq-fraction-prior): with a background, its fraction f₀ and each component's f₁, …, f_K (the parameters in that order); without one, the components' shares. One `alpha` is the same for every fraction. Each must be at least 1 (below 1 the density grows without bound as a fraction goes to 0). It acts like alphaₖ − 1 extra localizations from the background and from each component (@eq-fraction-map), and keeps every fraction above 0, so that no component can be switched off. On the signal fraction π = 1 − f₀ and the shares it is a Beta density of π, with parameters Σₖ₌₁ᴷ(alphaₖ − 1) + 1 and alpha0, and the Dirichlet density of the shares with alpha1, …, alphaK (@eq-fraction-split). The default is `Dirichlet(2, 5, …, 5)` with a background (for one component the Beta(5, 2) density of π, mode 0.8; for two, as 8 extra signal localizations and 1 background one, a pull toward π = 8/9 that grows with the number of components), and `Dirichlet(5)` without.
"""
struct Dirichlet{A<:Union{Real,Tuple{Vararg{Real}}}} <: Prior
    alpha::A
    function Dirichlet(alpha::Real...)
        isempty(alpha) && throw(ArgumentError("Dirichlet: alpha is missing"))
        all(>=(1), alpha) || throw(
            ArgumentError(
                "Dirichlet: every alpha must be at least 1, got $alpha: below 1 the density grows without bound as a fraction goes to 0",
            ),
        )
        a = length(alpha) == 1 ? only(alpha) : promote(alpha...)
        return new{typeof(a)}(a)
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
    Bingham(terms::Watson...)

The Bingham density of an axis on the sphere, ∝ exp(Σⱼ kappaⱼ (μⱼ·n)²): the product of the Watson densities `terms`, whose exponents it adds (derivations/derivation.qmd, @eq-bingham). With perpendicular μ's it holds the axis to a different spread in each direction: `Bingham(Watson(0, 0, -1000), Watson(π/2, -π/2, 2.6))` keeps an axis near the xy plane (its tilt out of it with SD ≈ 1/√(2·1000) rad ≈ 1.3°) and near ±y within it (its turn with SD ≈ 1/√(2·2.6) rad ≈ 25°). It gives n and −n the same density and is proper for every kappa. For component k > 1 of a composite, the μ's are in component 1's frame, as the axis is.
"""
struct Bingham{W<:Tuple{Vararg{Watson}}} <: Prior
    terms::W
    function Bingham(terms::Watson...)
        isempty(terms) &&
            throw(ArgumentError("a Bingham density needs at least one Watson term"))
        return new{typeof(terms)}(terms)
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

What is known about one quantity of a fit before it: its starting `value` (`nothing` to estimate it from the localizations, or else start from its default) and its `prior` (a [`Prior`](@ref); `nothing` for the quantity's default prior). Both are in the quantity's own terms: a position or a size in nm, a cap's `kappa` (κ′), an axis as `(theta, phi)`, the fractions as a tuple. [`run_locmofit`](@ref) lists the quantities and their default priors.
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
# a cap's curvature κ′ = sign(s)·sin(α/2), with the half-angle α the parameter n and s the parameter s: the cap of scale -s is the cap of scale s with the opposite curvature
struct Curvature{n,s} <: Quantity end
# the axis of component k, from its chart coordinates vx, vy: as m = normalize((vx, vy, 1)), the axis in the chart's frame
struct Axis{k,vx,vy} <: Quantity end
# the log fractions of the mixture, log softmax(ω₀, 0, ω₂, …) with a background and log softmax(0, ω₂, …) without (derivations/derivation.qmd, @eq-logfractions), with the parameters ω₂, … named `names`
struct Fractions{background,names} <: Quantity end

quantity(::Location{n}, θ) where {n} = getfield(θ, n)
quantity(::Size{n}, θ) where {n} = abs(getfield(θ, n))
quantity(::Curvature{n,s}, θ) where {n,s} = sign(getfield(θ, s)) * sin(getfield(θ, n) / 2)
function quantity(::Axis{k,vx,vy}, θ) where {k,vx,vy}
    a, b = getfield(θ, vx), getfield(θ, vy)
    r = inv(sqrt(1 + a^2 + b^2))
    return (a * r, b * r, r)
end
function quantity(::Fractions{background,names}, θ) where {background,names}
    lw = (
        (background ? (θ.omega0,) : ())...,
        zero(θ.sigma),
        ntuple(i -> getfield(θ, names[i]), Val(length(names)))...,
    )
    return lw .- logsumexp(lw)
end

# A Watson or Bingham density in the chart's coordinates, the quadratic form mᵀA′m with A′ = R₀ᵀAR₀ for the axis n = R₀m and A = Σⱼ κⱼμⱼμⱼᵀ, fixed for the whole fit (derivations/derivation.qmd, Axes): its six distinct entries a₁₁, a₂₂, a₃₃, a₁₂, a₁₃, a₂₃
struct ChartQuadratic{T} <: Prior
    a::NTuple{6,T}
end

# the log densities, up to a constant
log_density(p::Normal, q) = -((q - p.mean) / p.sd)^2 / 2
function log_density(p::LogNormal, q)
    l = log(q)
    return -l - ((l - p.mu) / p.sigma)^2 / 2
end
log_density(p::Dirichlet, logf::Tuple) = sum((p.alpha .- 1) .* logf)
function log_density(p::ChartQuadratic, m)
    a = p.a
    return a[1] * m[1]^2 +
           a[2] * m[2]^2 +
           a[3] * m[3]^2 +
           2 * (a[4] * m[1] * m[2] + a[5] * m[1] * m[3] + a[6] * m[2] * m[3])
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
supports(::Axis, ::Union{Watson,Bingham}) = true
supports(::Fractions, ::Dirichlet) = true
supports(::Quantity, ::Prior) = false

describe(::Location) = "position"
describe(::Size) = "size"
describe(::Curvature) = "curvature"
describe(::Axis) = "axis"
describe(::Fractions) = "set of fractions"

# the parameters a quantity is computed from
parameter_names(::Location{n}) where {n} = (n,)
parameter_names(::Size{n}) where {n} = (n,)
parameter_names(::Curvature{n}) where {n} = (n,)
parameter_names(::Axis{k,vx,vy}) where {k,vx,vy} = (vx, vy)
function parameter_names(::Fractions{background,names}) where {background,names}
    return background ? (:omega0, names...) : names
end

# A quantity's value as starting values of the parameters; `axes` are the names that set each component's starting axis (orientation_names)
start_values(q::Union{Location,Size}, v::Real, axes) = (only(parameter_names(q)) => v,)
function start_values(q::Curvature, v::Real, axes)
    abs(v) < 1 || throw(ArgumentError("kappa must lie in (-1, 1), got $v"))
    return (only(parameter_names(q)) => 2asin(v),)
end
function start_values(::Axis{k}, v, axes) where {k}
    length(v) == 2 ||
        throw(ArgumentError("an axis is given as (theta, phi), got $(repr(v))"))
    return (axes[k][1] => v[1], axes[k][2] => v[2])
end
# ωₖ = log(fₖ/f₁), with f₁ the fraction of component 1 (derivations/derivation.qmd, @eq-softmax)
function start_values(q::Fractions{background}, v, axes) where {background}
    n = nfractions(q)
    length(v) == n && all(>(0), v) || throw(
        ArgumentError("the fractions are given as $n positive numbers, got $(repr(v))")
    )
    ref = background ? 2 : 1
    others = Tuple(i for i in 1:n if i != ref)
    return Tuple(n => log(v[i]) - log(v[ref]) for (n, i) in zip(parameter_names(q), others))
end
function start_values(q::Quantity, v, axes)
    return throw(ArgumentError("a $(describe(q)) is not given as $(repr(v))"))
end

const SIZE_PRIOR = LogNormal(log(50.0) + 1, 1.0)
# the default Dirichlet's alpha for the background's fraction and for each component's
const BACKGROUND_ALPHA = 2.0
const COMPONENT_ALPHA = 5.0

function nfractions(::Fractions{background,names}) where {background,names}
    return length(names) + 1 + background
end

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

The quantities of a fit of `model`, each as `name => (quantity, default prior)`: those of every component (`quantities(component, suffix, index)`), the blur `sigma`, and the `fractions` of the mixture (with a background or several components).
"""
function quantities(m::SimpleModel, background::Bool)
    return (quantities(m, "", 1)..., common_quantities(m, background)...)
end
function common_quantities(m::Model, background)
    return (
        :sigma => (Size{:sigma}(), Flat()),
        fraction_quantities(share_names(m), background)...,
    )
end
# the fractions of the background (if any) and of the components (ω₂, … named `names`); none for a single component without a background
function fraction_quantities(names, background)
    !background && isempty(names) && return ()
    prior = if background
        Dirichlet(BACKGROUND_ALPHA, ntuple(_ -> COMPONENT_ALPHA, length(names) + 1)...)
    else
        Dirichlet(COMPONENT_ALPHA)
    end
    return (:fractions => (Fractions{background,names}(), prior),)
end
# the names of the parameters ω₂, …, ω_K of the components' fractions
share_names(::SimpleModel) = ()

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

The priors of [`specification`](@ref) as the tuple of [`PriorTerm`](@ref)s the loss evaluates, in the floating-point type `T`. A Watson or Bingham prior on an axis becomes a quadratic form in the chart's coordinates, which needs the chart's frame from the unit model.
"""
function prior_terms(priors, unit, T)
    return Tuple(prior_term(q, p, unit, T) for (q, p) in priors)
end
prior_term(q, p::Normal, unit, T) = PriorTerm(q, Normal(T(p.mean), T(p.sd)))
prior_term(q, p::LogNormal, unit, T) = PriorTerm(q, LogNormal(T(p.mu), T(p.sigma)))
function prior_term(q::Fractions, p::Dirichlet, unit, T)
    p.alpha isa Real ||
        length(p.alpha) == nfractions(q) ||
        throw(
            ArgumentError(
                "the Dirichlet prior has $(length(p.alpha)) alphas for $(nfractions(q)) fractions",
            ),
        )
    return PriorTerm(q, Dirichlet(T.(p.alpha)...))
end
prior_term(q::Axis, p::Watson, unit, T) = prior_term(q, Bingham(p), unit, T)
function prior_term(q::Axis{k}, p::Bingham, unit, T) where {k}
    R₀ = unit isa Tuple ? unit[k].chart : unit.chart
    A = sum(
        t.kappa *
        (R₀' * axis_vector(t.theta, t.phi)) *
        (R₀' * axis_vector(t.theta, t.phi))' for t in p.terms
    )
    return PriorTerm(
        q, ChartQuadratic(T.((A[1, 1], A[2, 2], A[3, 3], A[1, 2], A[1, 3], A[2, 3])))
    )
end
