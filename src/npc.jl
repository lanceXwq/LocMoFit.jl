struct NuclearPoreComplex <: Model end

initialparameters(::NuclearPoreComplex, dtype::Type{<:AbstractFloat}) = (
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
)

function unit_model(::NuclearPoreComplex, ps; spacing::Real)
    circs = 2π * ps.r1 .* (one(ps.r2), ps.r2)
    npoints = map(circ -> ceil(Int, circ / spacing), circs)
    return map(npoints -> unit_ring(eltype(ps[1]), npoints), npoints)
end

unit_model(::NuclearPoreComplex, ps, pts_per_ring::Tuple{<:Integer,<:Integer}) = map(pts -> unit_ring(eltype(ps[1]), pts), pts_per_ring)

LuxCore.initialstates(::AbstractRNG, ::NuclearPoreComplex) = NamedTuple()
LuxCore.parameterlength(::NuclearPoreComplex) = 12
LuxCore.statelength(::NuclearPoreComplex) = 0

function (::NuclearPoreComplex)(unit_npc, ps, st)
    fullps = fullparams(ps, st)
    npts = map(u -> size(u, 2), unit_npc)
    X = hcat(unit_npc...)
    X .*= fullps.r1
    X2 = view(X, :, npts[1] .+ (1:npts[2]))
    X2 .*= fullps.r2
    rot_trans!(X2, fullps.θ2, fullps.ϕ2, fullps.x2, fullps.y2, fullps.z2)
    rot_trans!(X, fullps.θ1, fullps.ϕ1, fullps.x1, fullps.y1, fullps.z1)
    return X, st
end