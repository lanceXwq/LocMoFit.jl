struct SpCapRing <: Model end

initialparameters(::SpCapRing, dtype::Type{<:AbstractFloat}) = (
    x1=convert(dtype, 0.0),
    y1=convert(dtype, 0.0),
    z1=convert(dtype, 0.0),
    r1=convert(dtype, 50), # 50 nm radius
    θ1=convert(dtype, 0.0),
    ϕ1=convert(dtype, 0.0),
    x2=convert(dtype, 0.0), # Δx relative to ring 1
    y2=convert(dtype, 0.0), # Δy relative to ring 1
    z2=convert(dtype, 50), # Δz relative to ring 1
    c2=convert(dtype, 1.0), # c2 / r1 is the curvature of the spherical cap
    α2=convert(dtype, π / 2),
    θ2=convert(dtype, 0.0), # orientation relative to ring 1
    ϕ2=convert(dtype, 0.0),
    σ=convert(dtype, 0.0),
)

function unit_model(::SpCapRing, ps; spacing::Real)
    ring_circumference = 2π * ps.r1
    cap_area = 4π / (ps.c2 / ps.r1)^2
    npoints_ring = ceil(Int, ring_circumference / spacing)
    npoints_cap = ceil(Int, cap_area / spacing^2)
    return (unit_ring(eltype(ps[1]), npoints_ring), unit_sphere(eltype(ps[1]), npoints_cap))
end

function unit_model(::SpCapRing, ps, npoints_per_shape::Tuple{<:Integer,<:Integer})
    T = eltype(ps[1])
    return (unit_ring(T, npoints_per_shape[1]), unit_sphere(T, npoints_per_shape[2]))
end

LuxCore.initialstates(::AbstractRNG, ::SpCapRing) = NamedTuple()
LuxCore.parameterlength(::SpCapRing) = 14
LuxCore.statelength(::SpCapRing) = 0

function (::SpCapRing)(unit_model, ps, st)
    fullps = fullparams(ps, st)
    n1 = size(unit_model[1], 2)
    n2 = size(unit_model[2], 2)
    # ns = map(u -> size(u, 2), unit_npc)
    u1 = copy(unit_model[1])
    u2 = unit_sphere_to_cap(copy(unit_model[2]), fullps.α2)
    X = hcat(u1, u2)
    scale!(X, fullps.r1)
    @views begin
        X2 = X[:, n1+1:n1+n2]
        scale!(X2, 1 / fullps.c2)
        rot_trans!(X2, fullps.θ2, fullps.ϕ2, fullps.x2, fullps.y2, fullps.z2)
    end
    rot_trans!(X, fullps.θ1, fullps.ϕ1, fullps.x1, fullps.y1, fullps.z1)
    return X, st
end

function penalty(::Union{T, Lux.Experimental.FrozenLayer{<:Any,T}}, ps, st) where {T<:SpCapRing}
    fullps = fullparams(ps, st)
    penalty_c1 = sq_hinge(1 / fullps.r1^2 - 1)
    penalty_c2 = sq_hinge(fullps.c2^2 - 1)
    penalty_θ1 = sq_hinge(fullps.θ1 - π) + sq_hinge(-fullps.θ1)
    penalty_θ2 = sq_hinge(fullps.θ2 - π) + sq_hinge(-fullps.θ2)
    return penalty_c1 + penalty_c2 + penalty_θ1 + penalty_θ2
end