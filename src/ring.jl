struct Ring <: Model end

initialparameters(::Ring, dtype::Type{<:AbstractFloat}) = (
    x=convert(dtype, 0.0),
    y=convert(dtype, 0.0),
    z=convert(dtype, 0.0),
    r=convert(dtype, 50), # 50 nm radius
    θ=convert(dtype, 0.0),
    ϕ=convert(dtype, 0.0),
    σ=convert(dtype, 0.0),
)

function unit_model(::Ring, ps; spacing::Real)
    circumference = 2π * ps.r
    npoints = ceil(Int, circumference / spacing)
    return unit_ring(eltype(ps[1]), npoints)
end

unit_model(::Ring, ps, npoints::Integer) = unit_ring(eltype(ps[1]), npoints)

LuxCore.initialstates(::AbstractRNG, ::Ring) = NamedTuple()
LuxCore.parameterlength(::Ring) = 7
LuxCore.statelength(::Ring) = 0

function (::Ring)(unit_ring, ps, st)
    fullps = fullparams(ps, st)
    # X = unit_ring_to_arc(unit_ring)
    X = scale!(copy(unit_ring), fullps.r)
    # X[3, :] .-= r
    rot_trans!(X, fullps.θ, fullps.ϕ, fullps.x, fullps.y, fullps.z)
    # rot_trans!(X, fullps.θ, fullps.ϕ, fullps.apex)
    return X, st
end

function penalty(::Ring, ps, st)
    fullps = fullparams(ps, st)
    penalty_c = sq_hinge(1 / fullps.r^2 - 1)
    penalty_θ = sq_hinge(fullps.θ - π) + sq_hinge(-fullps.θ)
    return penalty_c + penalty_θ
end