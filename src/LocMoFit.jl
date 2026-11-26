module LocMoFit

using LinearAlgebra
using ComponentArrays: ComponentVector

using Enzyme
using Enzyme: EnzymeRules, Annotation
using Optimization
using OptimizationOptimJL

include("models.jl")
include("priors.jl")
include("parameters.jl")
include("unit models.jl")
include("transformations.jl")
include("spcap.jl")
include("ring.jl")
include("composite.jl")
include("start.jl")
include("loss.jl")
include("rules.jl")

export Ring, SphericalCap, Composite, DoubleRing, SpCapRing, run_locmofit
export ParamToFit, Flat, Normal, LogNormal, Dirichlet, Watson, Bingham, Fixed, Tied

# BFGS rather than L-BFGS: with a few tens of parameters its n×n matrix costs nothing, and it is insensitive to the parameters' different scales (nm, radians, 1/nm), which DynMoFit found to matter for L-BFGS (DynMoFit.jl/DESIGN.md)
const BFGS_BACKTRACKING = BFGS(; linesearch=Optim.LineSearches.BackTracking())

"""
    fit_problem(model, locs, loc_precisions; params, spacing, point_params, dtype, background, fov_volume) -> (p, u₀)

Everything a fit needs: `p` for [`loss`](@ref) and the free parameters `u₀` to start from, with the starting values `params` leaves open estimated from the localizations ([`estimate_start`](@ref)). The keywords are those of [`run_locmofit`](@ref).
"""
function fit_problem(
    model::Model,
    locs::AbstractMatrix{<:Real},
    loc_precisions::AbstractMatrix{<:Real};
    params::NamedTuple=NamedTuple(),
    spacing::Real=3.0,
    point_params::NamedTuple=NamedTuple(),
    dtype::Type{<:AbstractFloat}=Float64,
    background::Bool=true,
    fov_volume::Union{Nothing,Real}=nothing,
)
    size(loc_precisions) == size(locs) || throw(
        DimensionMismatch(
            "loc_precisions must match locs shape; got $(size(loc_precisions)) vs $(size(locs))",
        ),
    )

    # the starting axes, as theta, phi (thetak, phik for component k), center the orientation charts; they are not parameters
    axes = orientation_names(model)
    axis_names = Tuple(Iterators.flatten(axes))
    # params in the optimizer's parameters: starting values, fixed parameters, ties, and priors
    spec = specification(model, background, params, axes)
    given, fixed, tie = spec.init, spec.fixed, spec.tie
    # what params leaves open, fixed and tied parameters excepted, starts from the data (derivations/derivations.qmd), along the starting axis of component 1 if it is given or held
    θa, ϕa = axes[1]
    given_axis =
        haskey(given, θa) || haskey(given, ϕa) || any(k -> k in fixed, chart_names(model))
    axis = given_axis ? axis_vector(get(given, θa, 0), get(given, ϕa, 0)) : nothing
    targets = map(first, tie)
    isopen(k) = !haskey(given, k) && !(k in fixed) && !(k in targets)
    estimate = if any(isopen, start_names(model, size(locs, 1)))
        estimate_start(model, locs, loc_precisions, axis)
    else
        NamedTuple()
    end
    open = Tuple(k for k in keys(estimate) if isopen(k))
    init = merge(NamedTuple{open}(estimate), given)
    starts = map(
        n -> (convert(dtype, get(init, n[1], 0)), convert(dtype, get(init, n[2], 0))), axes
    )

    θ₀ = default_parameters(model, dtype)
    background && (θ₀ = merge(θ₀, (omega0=zero(dtype),)))
    θ₀ = apply_init(θ₀, Base.structdiff(init, NamedTuple{axis_names}))
    # a background fraction not given starts at 0.1, whatever the components' shares: ω₀ = log(f₀/f₁) with f₁ = 0.9w₁ (derivations/derivations.qmd, @eq-softmax)
    if background && !haskey(init, :omega0)
        θ₀ = merge(θ₀, (omega0=log(dtype(1 / 9)) + logsumexp(log_weights(model, θ₀)),))
    end
    pmap = ParameterMap(θ₀, fixed, tie)
    θ₀ = full_parameters(free_parameters(θ₀, pmap), pmap)      # with the ties applied
    # the values the points are generated from: point_params in place of the starting values, with the ties applied
    θpoints = if isempty(point_params)
        θ₀
    else
        θp = apply_init(θ₀, point_params)
        mp = ParameterMap(θp, fixed, tie)
        full_parameters(free_parameters(θp, mp), mp)
    end
    unit = unit_model(model, θpoints, starts; spacing)

    locsT = locs isa Array{dtype} ? locs : Array{dtype}(locs)
    V = if fov_volume === nothing
        prod(maximum(locsT; dims=2) .- minimum(locsT; dims=2))
    else
        fov_volume
    end
    p = (
        model=model,
        parameter_map=pmap,
        unit_model=unit,
        ranges=point_ranges(unit),
        locs=locsT,
        loc_prec=if loc_precisions isa Array{dtype}
            loc_precisions
        else
            Array{dtype}(loc_precisions)
        end,
        background=background,
        log_volume=convert(dtype, log(V)),
        priors=prior_terms(spec.priors, unit, dtype),
    )
    return p, free_parameters(θ₀, pmap)
end

# the names that set each component's starting axis
orientation_names(::SimpleModel) = ((:theta, :phi),)
function orientation_names(c::Composite)
    return ntuple(k -> (Symbol(:theta, k), Symbol(:phi, k)), ncomponents(c))
end
# the chart coordinates of component 1's axis
chart_names(::SimpleModel) = (:vx, :vy)
chart_names(::Composite) = (:vx1, :vy1)

"""
    unit_model(model, θ, starts; spacing)

What a model is built from, made once from the starting values θ: the unit points and the frame of the orientation's chart, `rotmat(θ₀, ϕ₀)` of the starting axis (`starts`, one `(θ₀, ϕ₀)` per component), as `(points, chart)`, or a tuple of them for a composite.
"""
function unit_model(m::SimpleModel, θ, starts; spacing)
    return (points=unit_points(m, θ; spacing), chart=rotmat(starts[1]...))
end
function unit_model(c::Composite, θ, starts; spacing)
    return map(
        (points, start) -> (points=points, chart=rotmat(start...)),
        unit_points(c, θ; spacing),
        starts,
    )
end

# the rows of each component's points among all model points (the components' points are concatenated in order)
point_ranges(unit::NamedTuple) = (1:size(unit.points, 2),)
function point_ranges(unit::Tuple)
    return (
        n=map(u -> size(u.points, 2), unit);
        ends=cumsum(n);
        map((e, k) -> (e - k + 1):e, ends, n)
    )
end

"""
    components(model, unit_model, θ)

Each component in the world frame: the position of its reference point (a ring's center, a cap's apex), its axis as a unit vector and as polar angle `theta` and azimuth `phi`, and its shape in the usual terms (a ring's radius `r`; a cap's scale `s`, dimensionless curvature `kappa` (κ′ = sin(α/2) of the fitted half-angle α), `curvature` (κ′/s), sphere `radius` (s/κ′), half-angle `alpha` (in [0, π], from κ′: acos(1 − 2κ′²)) and `area`).
"""
function components(m::SimpleModel, unit_model, θ)
    R = chart_rotation(unit_model.chart, θ.vx, θ.vy)
    return (merge(pose_summary([θ.x, θ.y, θ.z], R[:, 3]), shape_summary(m, θ)),)
end
function components(c::Composite, unit_model, θ)
    p1 = component_parameters(c, θ, Val(1))
    R1, t1 = chart_rotation(unit_model[1].chart, p1.vx, p1.vy), [p1.x, p1.y, p1.z]
    return ntuple(ncomponents(c)) do k
        pk = component_parameters(c, θ, Val(k))
        Rk = k == 1 ? R1 : R1 * chart_rotation(unit_model[k].chart, pk.vx, pk.vy)
        tk = k == 1 ? t1 : t1 + R1 * [pk.x, pk.y, pk.z]
        return merge(pose_summary(tk, Rk[:, 3]), shape_summary(c.components[k], pk))
    end
end
function pose_summary(position, n)
    return (position=position, axis=n, theta=acos(clamp(n[3], -1, 1)), phi=atan(n[2], n[1]))
end

"""
    objective(p, u₀)

The `OptimizationFunction` of a fit: [`loss`](@ref) and its Enzyme gradient. The line search asks for the value at a trial point and, once it accepts the point (nearly always the first one), for the gradient there. A gradient comes with its value and costs less than two values, so every value is computed with its gradient, which is kept for the call that follows.
"""
function objective(p, u₀)
    u_last, g_last, v_last = fill!(similar(u₀), NaN), zero(u₀), Ref(zero(eltype(u₀)))
    function value(u, _)
        u == u_last && return v_last[]
        fill!(g_last, 0)
        _, v_last[] = Enzyme.autodiff(
            ReverseWithPrimal, loss, Active, Duplicated(u, g_last), Const(p)
        )
        copyto!(u_last, u)
        return v_last[]
    end
    function gradient!(G, u, _)
        value(u, nothing)
        copyto!(G, g_last)
        return nothing
    end
    function value_gradient!(G, u, _)
        v = value(u, nothing)
        copyto!(G, g_last)
        return v
    end
    return OptimizationFunction(value; grad=gradient!, fg=value_gradient!)
end

"""
    run_locmofit(model, locs, loc_precisions; params, max_iter, tol, spacing, point_params, dtype, background,
                 fov_volume, history)

Fit `model` to one site's localizations by maximizing the posterior of derivations/derivations.qmd: a mixture of the structure and a uniform background, with a prior on every quantity (@eq-full-posterior), by BFGS with a backtracking line search (gradients by Enzyme). `model` is a [`Model`](@ref), such as `Ring()` or `Composite(Ring(), SphericalCap())`, or the name of a function in LocMoFit that returns one, such as `"Ring"` or `"DoubleRing"`. A fit runs on one thread. Fitting many sites in threads (`Threads.@threads`) is much faster, but has rarely crashed Julia with memory corruption, for reasons not yet known; separate processes are safe.

- `locs`, `loc_precisions`: `D×N` positions and per-axis precisions (standard deviations), in nm, with `D ∈ {2, 3}`. A 2D fit uses the first two coordinates of the model.
- `params`: a `NamedTuple` of [`ParamToFit`](@ref), the starting value and the prior of each quantity it names, both in the quantity's own terms; an unknown name throws. A component k of a composite has its number appended to its quantities' names (`x1`, `r1`, `axis2`, …). The quantities and their default priors:
    - `x`, `y`, `z`: the position of the reference point (a ring's center, a cap's apex), in nm; for component k > 1, in component 1's frame. `Flat()`.
    - `r` (`Ring`), `s` (`SphericalCap`): the size, in nm (a cap's area is 4πs²). `LogNormal(log(50) + 1, 1)`: most probable at 50 nm, median 136 nm, 95% between 19 and 965 nm.
    - `kappa` (`SphericalCap`): the dimensionless curvature κ′ in [−1, 1], from a flat disc at 0 to a whole sphere at ±1 (a starting value inside (−1, 1)). `Flat()`. It is fitted as the cap's half-angle `alpha`, with κ′ = sin(`alpha`/2).
    - `axis`: the axis, as `(theta, phi)`, its polar angle and azimuth; for component k > 1, in component 1's frame. `Flat()`, uniform on the sphere; it takes a `Watson` or `Bingham` density. It is fitted as the coordinates `vx`, `vy` of a tangent chart centered on its starting value.
    - `sigma`: the blur, in nm, added to every precision in quadrature. `Flat()`.
    - `fractions` (with a background or several components): the fractions of the localizations from the background and from each component, as a tuple `(f0, f1, …, fK)` (without a background `(f1, …, fK)`, the components' shares), positive numbers of which only the ratios count. `Dirichlet(2, 5, …, 5)` (`Dirichlet(5)` without a background): as much as 1 background localization and 4 from each component, so that the background and every component keep some of the localizations and no component can be switched off. On the signal fraction π = 1 − f0 it acts as K·4 signal localizations and 1 background one, so that for one component it is the Beta(5, 2) density of π (mode 0.8) and for two pulls π toward 8/9; it moves a component's share w of the signal by K(α − 1)(1/K − w)/(n + K(α − 1)) for n signal localizations and α = 5 (with two components and 100 signal localizations, a share of 0.3 becomes 0.315).
  `Fixed()` holds a quantity at its value and `Tied(:r1)` at another quantity's value throughout (derivations/derivations.qmd, Fixed and tied values). A starting value not given is estimated from the localizations ([`estimate_start`](@ref); derivations/derivations.qmd, Starting values from the data), unless the quantity is fixed or tied: then, and wherever nothing is estimated, it starts from its default (`default_parameters`, plus a background fraction of 0.1 with a background). The estimate covers a `Ring`, a `SphericalCap` and a composite of two of them sharing an axis (such as `DoubleRing` and `SpCapRing`): in 3D the position, the axis and the shape, and for a composite the second component's distance along the axis (`z2`) and the shares; in 2D only the position. Any other model gets only the position of component 1, at the weighted center of the localizations. The blur and the background fraction start from their defaults, and an axis not estimated from (0, 0), along z. An axis given for component 1, or held, is used instead of the estimated one, and the shape is estimated along it.
- `max_iter`: the most iterations the optimizer takes.
- `tol`: the fit has converged when no component of the gradient of [`loss`](@ref) exceeds `tol`. The default, 1e-4, leaves the parameters much closer to the optimum than their statistical error (a position within about 0.005 nm for a ring of 100 localizations, whose error is about 0.7 nm).
- `spacing`: the distance between model points, in nm. The points are generated once, from the starting values, and stay fixed during the fit.
- `point_params`: values that replace the starting values only for generating the points (with the ties applied), such as `(r=50,)`: then the number of points comes from a ring of radius 50 nm, whatever the starting radius. Only the shape parameters that set the number of points matter: a ring's `r` and a cap's `s` (`r1`, `s2`, … in a composite). Unknown names are ignored with a warning.
- `dtype`: the floating-point type of the fit, `Float64` by default. `Float32` is faster on large models (a cap fit 1.6×) but can stop early, before the gradient reaches `tol` (a known issue: DESIGN.md).
- `background`: whether some localizations are background, uniform over the field of view; their share is f0 of `fractions`.
- `fov_volume`: the volume (area, in 2D) of the field of view the localizations were selected from, in nm³ (nm²). Without it, the bounding box of the localizations is used, which is a little smaller.
- `history`: also record the loss and the parameters after every iteration (for debugging and videos of the fit).

Returns a `NamedTuple` with:

- `parameters`: all of the optimizer's parameters, held and tied ones included, as a `NamedTuple` (`omega0`, `omega2`, … for the fractions, ωₖ = log(fₖ/f₁), `alpha` for a cap's half-angle, κ′ = sin(`alpha`/2), which may come out negative or beyond π, where the shapes repeat (`components` reports the half-angle in [0, π]), `vx`, `vy` for an axis);
- `signal_fraction`: 1 − f0, or `nothing` without a background;
- `weights`: the components' shares of the signal localizations (one share, 1, for a simple model);
- `components`: each component in the world frame: its position, its axis (`axis`, `theta`, `phi`), and its shape (see [`components`](@ref));
- `loss`: the final value of [`loss`](@ref);
- `model`, `unit_model`: the model and what it is built from (unit points, chart frames), which `model_points(model, unit_model, θ)` turns into model points for any parameters θ;
- `model_points`: the fitted model points (`3×M`);
- `result`: the Optimization.jl solution (return code, statistics);
- `history`: `nothing`, unless `history = true`; then `(losses, parameters)`, one entry per iteration.
"""
function run_locmofit(
    model_name::AbstractString,
    locs::AbstractMatrix{<:Real},
    loc_precisions::AbstractMatrix{<:Real};
    kwargs...,
)
    return run_locmofit(build_model(model_name), locs, loc_precisions; kwargs...)
end

function run_locmofit(
    model::Model,
    locs::AbstractMatrix{<:Real},
    loc_precisions::AbstractMatrix{<:Real};
    max_iter::Integer=200,
    tol::Real=1e-4,
    history::Bool=false,
    kwargs...,
)
    p, u₀ = fit_problem(model, locs, loc_precisions; kwargs...)
    pmap = p.parameter_map
    prob = OptimizationProblem(objective(p, u₀), u₀, p)

    if history
        losses = eltype(u₀)[]
        parameters = typeof(full_parameters(u₀, pmap))[]
        function callback(state, l)
            push!(losses, l)
            push!(parameters, full_parameters(state.u, pmap))
            return false
        end
        result = solve(prob, BFGS_BACKTRACKING; callback, maxiters=max_iter, g_abstol=tol)
    else
        result = solve(prob, BFGS_BACKTRACKING; maxiters=max_iter, g_abstol=tol)
    end

    θ = full_parameters(result.u, pmap)
    lw = log_weights(model, θ)
    lse = logsumexp(lw)
    return (
        parameters=θ,
        signal_fraction=p.background ? exp(lse - logaddexp(θ.omega0, lse)) : nothing,
        weights=exp.(lw .- lse),
        components=components(model, p.unit_model, θ),
        loss=result.objective,
        model=model,
        unit_model=p.unit_model,
        model_points=model_points(model, p.unit_model, θ),
        result=result,
        history=history ? (losses=losses, parameters=parameters) : nothing,
    )
end

end
