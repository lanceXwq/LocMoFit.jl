struct DoubleRing <: Model end

initialparameters(::DoubleRing, dtype::Type{<:AbstractFloat}) = (
    x1=convert(dtype, 0.0),
    y1=convert(dtype, 0.0),
    z1=convert(dtype, 0.0),
    r1=convert(dtype, 50), # 50 nm radius
    θ1=convert(dtype, 0.0),
    ϕ1=convert(dtype, 0.0),
    x2=convert(dtype, 0.0), # Δx relative to ring 1
    y2=convert(dtype, 0.0), # Δy relative to ring 1
    z2=convert(dtype, 50), # Δz relative to ring 1
    r2=convert(dtype, 1), # multiplicative factor relative to r1
    θ2=convert(dtype, 0.0), # orientation relative to ring 1
    ϕ2=convert(dtype, 0.0),
    σ=convert(dtype, 0.0),
)

function unit_model(::DoubleRing, ps; spacing::Real)
    circumferences = 2π * ps.r1 .* (one(ps.r2), ps.r2)
    npts_per_shape = map(circ -> ceil(Int, circ / spacing), circumferences)
    return map(npoints -> unit_ring(eltype(ps[1]), npoints), npts_per_shape)
end

unit_model(::DoubleRing, ps, npoints_per_shape::Tuple{<:Integer,<:Integer}) = map(npoints -> unit_ring(eltype(ps[1]), npoints), npoints_per_shape)

LuxCore.initialstates(::AbstractRNG, ::DoubleRing) = NamedTuple()
LuxCore.parameterlength(::DoubleRing) = 13
LuxCore.statelength(::DoubleRing) = 0

function (::DoubleRing)(unit_model, ps, st)
    fullps = fullparams(ps, st)
    n1 = size(unit_model[1], 2)
    n2 = size(unit_model[2], 2)
    # ns = map(u -> size(u, 2), unit_npc)
    X = hcat(unit_model...)
    scale!(X, fullps.r1)
    @views begin
        X2 = X[:, n1+1:n1+n2]
        scale!(X2, fullps.r2)
        rot_trans!(X2, fullps.θ2, fullps.ϕ2, fullps.x2, fullps.y2, fullps.z2)
    end
    rot_trans!(X, fullps.θ1, fullps.ϕ1, fullps.x1, fullps.y1, fullps.z1)
    return X, st
end

function penalty(::DoubleRing, ps, st)
    fullps = fullparams(ps, st)
    penalty_c1 = sq_hinge(1 / fullps.r1^2 - 1)
    penalty_c2 = sq_hinge(1 / fullps.r2^2 - 1)
    penalty_θ1 = sq_hinge(fullps.θ1 - π) + sq_hinge(-fullps.θ1)
    penalty_θ2 = sq_hinge(fullps.θ2 - π) + sq_hinge(-fullps.θ2)
    return penalty_c1 + penalty_c2 + penalty_θ1 + penalty_θ2
end