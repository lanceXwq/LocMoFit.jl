"""
    Model

A geometric model: a [`SimpleModel`](@ref) (one shape) or a [`Composite`](@ref) of several. A model type only selects methods: its parameters are not stored in it but in the fitted vector (see [`ParameterMap`](@ref)). Every model has

- `default_parameters(model, T)`: a `NamedTuple` of scalars of type `T`, which gives the parameter names and their default starting values;
- `unit_points(model, θ; spacing)`: the points the model is built from (a matrix, or a tuple of matrices for a composite), generated once from the starting values θ, `spacing` nm apart;
- `model_points(model, unit_model, θ)`: the `3×M` model points for the parameters θ (a `NamedTuple`), without changing the unit model (`(points, chart)` from [`fit_problem`](@ref): the unit points and the frame of the orientation's chart, or a tuple of them for a composite);
- `quantities(model, background)`: the quantities that [`ParamToFit`](@ref) gives starting values and priors for, each computed from the parameters, with its default prior.

Every model has a blur `sigma`, which [`loss`](@ref) combines with the localization precisions.
"""
abstract type Model end

"""
    SimpleModel

A model of one shape. It defines only the shape, in its own frame (with its reference point, such as a ring's center or a cap's apex, at the origin and its axis along z):

- `shape_parameters(model, T)`: the shape's parameters and their defaults (e.g. a ring's radius `r`);
- `unit_points(model, shape; spacing)`, with `shape` holding (at least) the shape parameters;
- `local_points(model, unit_points, shape)`: the `3×M` points in the model's own frame;
- `shape_quantities(model, suffix)`: the shape's quantities, as `name => (quantity, default prior)` with `suffix` appended to the names (such as a ring's radius, `Size{:r}()` with the default prior on sizes);
- `shape_summary(model, shape)`: the shape's size and shape in the usual terms (e.g. a cap's radius and area), which [`components`](@ref) reports.

The pose places it: the position `x`, `y`, `z` of its reference point and its axis, given by the coordinates `vx`, `vy` of a tangent chart centered on the starting axis ([`chart_rotation`](@ref)): `vx = vy = 0` is the starting axis, the `axis` that `run_locmofit` takes as `(theta, phi)` (polar angle and azimuth) in `params`. Its parameters are `(x, y, z, shape..., vx, vy, sigma)`.
"""
abstract type SimpleModel <: Model end

# the blur starts at 1 nm: the likelihood depends on σ only through σ², so a fit started at σ = 0 never moves it
function default_parameters(m::SimpleModel, T::Type{<:AbstractFloat})
    return (
        x=zero(T),
        y=zero(T),
        z=zero(T),
        shape_parameters(m, T)...,
        vx=zero(T),
        vy=zero(T),
        sigma=one(T),
    )
end

# the parameter names of the shape and its pose, without the blur
local_names(m::SimpleModel) = (:x, :y, :z, keys(shape_parameters(m, Float64))..., :vx, :vy)

function model_points(m::SimpleModel, unit_model, θ)
    return place(
        local_points(m, unit_model.points, θ),
        chart_rotation(unit_model.chart, θ.vx, θ.vy),
        θ.x,
        θ.y,
        θ.z,
    )
end

# the log weights of the components, before normalization; a simple model has one component
log_weights(::SimpleModel, θ) = (zero(θ.sigma),)

build_model(model_name::AbstractString) = getfield(LocMoFit, Symbol(model_name))()

function apply_init(default::NamedTuple, init::NamedTuple)
    allowed = keys(default)
    init_keys = keys(init)
    isempty(init_keys) && return default

    unknown = Tuple(k for k in init_keys if !(k in allowed))
    if !isempty(unknown)
        @warn "Ignoring unknown init keys" unknown = unknown allowed = allowed
    end

    vals = ntuple(length(allowed)) do i
        k = allowed[i]
        old = getproperty(default, k)
        haskey(init, k) || return old
        new = getproperty(init, k)
        new isa Number ||
            throw(ArgumentError("init[$k] must be a Number; got $(typeof(new))"))
        return convert(typeof(old), new)
    end

    return NamedTuple{allowed}(vals)
end
