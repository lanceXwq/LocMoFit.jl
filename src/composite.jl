"""
    Composite(components...; defaults=NamedTuple())

Several simple models fitted together as one structure. Component 1 is placed in the world by its pose; every other component is placed by its pose in component 1's frame, so that its position and axis are relative to component 1.

Each component's parameters keep their names with the component's number appended (`x1`, `r1`, `x2`, `s2`, `eta2`, …); the blur `sigma` is shared. Shape parameters are absolute (a ring's radius and a cap's scale in nm): to make two of them equal, tie them (`tie=(:r2 => :r1,)` in [`run_locmofit`](@ref)). `defaults` overrides the generated default starting values.

Component k explains a share wₖ of the signal localizations, with its points averaged on their own. The shares are `softmax(0, omega2, …, omegaK)`, so the parameters `omega2, …, omegaK` (log wₖ/w₁) are fitted; they start at 0 (equal shares), and freezing them fixes the shares.
"""
struct Composite{C<:Tuple{Vararg{SimpleModel}},D<:NamedTuple,L} <: Model
    components::C
    defaults::D
end

# `L` holds each component's own parameter names, so that the generated functions below read them from the type instead of calling methods (which a generator may not do for methods defined after it, such as a user's model)
function Composite(components::SimpleModel...; defaults::NamedTuple=NamedTuple())
    return Composite{typeof(components),typeof(defaults),map(local_names, components)}(
        components, defaults
    )
end

ncomponents(::Composite{C}) where {C} = fieldcount(C)

function default_parameters(c::Composite, T::Type{<:AbstractFloat})
    parts = ntuple(ncomponents(c)) do k
        θk = default_parameters(c.components[k], T)
        names = local_names(c.components[k])
        return NamedTuple{map(n -> Symbol(n, k), names)}(map(n -> θk[n], names))
    end
    weights = NamedTuple{ntuple(k -> Symbol(:omega, k + 1), ncomponents(c) - 1)}(
        ntuple(_ -> zero(T), ncomponents(c) - 1)
    )
    return apply_init(merge(parts..., (sigma=one(T),), weights), c.defaults)
end

# The parameters of component k, under the component's own names: (x = θ.x2, y = θ.y2, …). Generated from the names in the type, so that it is plain field access (type-stable, and nothing for Enzyme to trip over).
@generated function component_parameters(
    ::Composite{C,D,L}, θ::NamedTuple, ::Val{k}
) where {C,D,L,k}
    return Expr(:tuple, [Expr(:(=), n, :(θ.$(Symbol(n, k)))) for n in L[k]]...)
end

function unit_points(c::Composite, θ; spacing::Real)
    return ntuple(
        k -> unit_points(c.components[k], component_parameters(c, θ, Val(k)); spacing),
        ncomponents(c),
    )
end

# every component k > 1 in component 1's frame; then all of them placed by component 1's pose
@generated function model_points(
    c::Composite{C}, unit_model::Tuple, θ::NamedTuple
) where {C}
    N = fieldcount(C)
    body = Expr[]
    for k in 1:N
        pk, Xk, Rk = Symbol(:p, k), Symbol(:X, k), Symbol(:R, k)
        push!(body, :($pk = component_parameters(c, θ, Val($k))))
        push!(body, :($Xk = local_points(c.components[$k], unit_model[$k].points, $pk)))
        push!(body, :($Rk = chart_rotation(unit_model[$k].chart, $pk.vx, $pk.vy)))
        k > 1 && push!(body, :($Xk = place($Xk, $Rk, $pk.x, $pk.y, $pk.z)))
    end
    push!(
        body,
        :(return place(hcat($((Symbol(:X, k) for k in 1:N)...)), R1, p1.x, p1.y, p1.z)),
    )
    return Expr(:block, body...)
end

# log w₁ = 0 and log wₖ/w₁ = ωₖ, before normalization
@generated log_weights(::Composite{C}, θ::NamedTuple) where {C} =
    Expr(:tuple, :(zero(θ.sigma)), (:(θ.$(Symbol(:omega, k))) for k in 2:fieldcount(C))...)

# every component's quantities, with its number appended to their names, then the shared ones
function quantities(c::Composite, background::Bool)
    parts = ntuple(k -> quantities(c.components[k], k, k), ncomponents(c))
    return (Iterators.flatten(parts)..., common_quantities(c, background)...)
end
function share_quantities(c::Composite)
    names = ntuple(k -> Symbol(:omega, k + 1), ncomponents(c) - 1)
    return (:shares => (Shares{names}(), SHARE_PRIOR),)
end

"""
    DoubleRing()

Two rings, the second one placed in the first one's frame: `Composite(Ring(), Ring())`, starting 50 nm apart along the axis. For the two rings of a nuclear pore, which have the same radius, tie the radii: `tie=(:r2 => :r1,)`.
"""
DoubleRing() = Composite(Ring(), Ring(); defaults=(z2=50,))

"""
    SpCapRing()

A ring and a spherical cap placed in the ring's frame: `Composite(Ring(), SphericalCap())`, with the cap's apex starting 100 nm along the ring's axis.
"""
SpCapRing() = Composite(Ring(), SphericalCap(); defaults=(z2=100,))
