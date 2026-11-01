module LocMoFit

using LinearAlgebra
using ComponentArrays: ComponentVector

using Enzyme
using Enzyme: EnzymeRules, Annotation
using Optimization
using OptimizationOptimJL

include("models.jl")
include("parameters.jl")
include("unit models.jl")
include("transformations.jl")
include("spcap.jl")
include("ring.jl")
include("composite.jl")
include("loss.jl")
include("rules.jl")

export Ring, SphericalCap, Composite, DoubleRing, SpCapRing, run_locmofit

# BFGS rather than L-BFGS: with a few tens of parameters its n×n matrix costs nothing, and it is insensitive to the parameters' different scales (nm, radians, 1/nm), which DynMoFit found to matter for L-BFGS (DynMoFit.jl/NOTES.md)
const BFGS_BACKTRACKING = BFGS(; linesearch=Optim.LineSearches.BackTracking())

"""
    fit_problem(model, locs, loc_precisions; init_params, freeze, tie, spacing, dtype, background, fov_volume,
                signal_prior) -> (p, u₀)

Everything a fit needs: `p` for [`loss`](@ref) and the free parameters `u₀` to start from. The keywords are those of [`run_locmofit`](@ref).
"""
function fit_problem(
    model::Model,
    locs::AbstractMatrix{<:Real},
    loc_precisions::AbstractMatrix{<:Real};
    init_params::NamedTuple=NamedTuple(),
    freeze=(),
    tie=(),
    spacing::Real=3.0,
    dtype::Type{<:AbstractFloat}=Float64,
    background::Bool=true,
    fov_volume::Union{Nothing,Real}=nothing,
    signal_prior::Union{Nothing,Tuple{Real,Real}}=(5, 2),
)
    size(loc_precisions) == size(locs) || throw(
        DimensionMismatch(
            "loc_precisions must match locs shape; got $(size(loc_precisions)) vs $(size(locs))",
        ),
    )

    # the starting axes, as theta, phi (thetak, phik for component k), center the orientation charts; they are not parameters
    axes = orientation_names(model)
    axis_names = Tuple(Iterators.flatten(axes))
    frozen_axes = Tuple(k for k in freeze_names(freeze) if k in axis_names)
    isempty(frozen_axes) || throw(
        ArgumentError(
            "theta and phi only set the starting axis, $frozen_axes; to keep the " *
            "orientation, freeze its chart coordinates (vx, vy, or vxk, vyk for component k)",
        ),
    )
    starts = map(
        n -> (
            convert(dtype, get(init_params, n[1], 0)),
            convert(dtype, get(init_params, n[2], 0)),
        ),
        axes,
    )

    θ₀ = default_parameters(model, dtype)
    background && (θ₀ = merge(θ₀, (logit_prob=convert(dtype, log(9)),)))   # signal fraction 0.9
    θ₀ = apply_init(θ₀, Base.structdiff(init_params, NamedTuple{axis_names}))
    pmap = ParameterMap(θ₀, freeze_names(freeze), tie_pairs(tie))
    θ₀ = full_parameters(free_parameters(θ₀, pmap), pmap)      # with the ties applied
    unit = unit_model(model, θ₀, starts; spacing)

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
        signal_prior=signal_prior === nothing ? nothing : convert.(dtype, signal_prior),
    )
    return p, free_parameters(θ₀, pmap)
end

# the names that set each component's starting axis
orientation_names(::SimpleModel) = ((:theta, :phi),)
function orientation_names(c::Composite)
    return ntuple(k -> (Symbol(:theta, k), Symbol(:phi, k)), ncomponents(c))
end

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

Each component in the world frame: the position of its reference point (a ring's center, a cap's apex), its axis as a unit vector and as polar angle `theta` and azimuth `phi`, and its shape in the usual terms (a ring's radius `r`; a cap's scale `s`, dimensionless curvature `kappa` (κ′ = tanh `eta`), `curvature` (κ′/s), sphere `radius` (s/κ′), half-angle `alpha` and `area`).
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
    run_locmofit(model, locs, loc_precisions; init_params, freeze, tie, max_iter, tol, spacing, dtype, background,
                 fov_volume, signal_prior, history)

Fit `model` to one site's localizations by maximizing the posterior of derivations/derivation.qmd: a mixture of the structure and a uniform background, with a Beta prior on the signal fraction, by BFGS with a backtracking line search (gradients by Enzyme). `model` is a [`Model`](@ref), such as `Ring()` or `Composite(Ring(), SphericalCap())`, or the name of a function in LocMoFit that returns one, such as `"Ring"` or `"DoubleRing"`. A fit runs on one thread. Fitting many sites in threads (`Threads.@threads`) is much faster, but has rarely crashed Julia with memory corruption, for reasons not yet known; separate processes are safe.

- `locs`, `loc_precisions`: `D×N` positions and per-axis precisions (standard deviations), in nm, with `D ∈ {2, 3}`. A 2D fit uses the first two coordinates of the model.
- `init_params`: starting values that replace the defaults (`default_parameters`, plus `logit_prob` with a background); unknown names are ignored with a warning. The starting axis is given as `theta`, `phi` (polar angle and azimuth; `thetak`, `phik` for component k, in component 1's frame for k > 1, default 0): the orientation is fitted as the coordinates `vx`, `vy` of a tangent chart centered on it, which start at 0.
- `freeze`: names of the parameters kept at their starting values (for the orientation, `vx` and `vy`).
- `tie`: pairs `target => source`, such as `:r2 => :r1`: the target takes the source's value throughout.
- `max_iter`: the most iterations the optimizer takes.
- `tol`: the fit has converged when no component of the gradient of [`loss`](@ref) exceeds `tol`. The default, 1e-4, leaves the parameters much closer to the optimum than their statistical error (a position within about 0.005 nm for a ring of 100 localizations, whose error is about 0.7 nm).
- `spacing`: the distance between model points, in nm. The points are generated once, from the starting values, and stay fixed during the fit.
- `dtype`: the floating-point type of the fit, `Float64` by default. `Float32` is faster on large models (a cap fit 1.6×) but can stop early, before the gradient reaches `tol` (a known issue: NOTES.md).
- `background`: whether some localizations are background, uniform over the field of view; their share is 1 − logistic(`logit_prob`), fitted from 0.1.
- `fov_volume`: the volume (area, in 2D) of the field of view the localizations were selected from, in nm³ (nm²). Without it, the bounding box of the localizations is used, which is a little smaller.
- `signal_prior`: `(a, b)` of the Beta prior on the signal fraction, or `nothing` for none. The default, (5, 2), has its mode at 0.8 and weighs as much as 5 localizations, so it keeps the fraction inside (0, 1) and pulls it toward 0.8 only on small sites.
- `history`: also record the loss and the parameters after every iteration (for debugging and videos of the fit).

Returns a `NamedTuple` with:

- `parameters`: all parameters, frozen and tied ones included, as a `NamedTuple`;
- `signal_fraction`: logistic(`logit_prob`), or `nothing` without a background;
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
    return (
        parameters=θ,
        signal_fraction=p.background ? 1 / (1 + exp(-θ.logit_prob)) : nothing,
        weights=exp.(lw .- logsumexp(lw)),
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
