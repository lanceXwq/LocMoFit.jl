struct NPCLayer{T} <: Lux.AbstractLuxLayer
    npoints::Int
    half_precision::T # precision here is the reciprocal of variance
    log_const::T
    NPCLayer(npoints=8, stddev::T) where {T} = new{T}(npoints, stddev^-2 / 2, -log(stddev^2 * 2π) / 2)
end

function (l::NPCLayer{T})(_, ps, st) where {T}
    angles = T.([2π / l.npoints * n for n in 0:(l.npoints-1)]) .+ mod(ps.offset_angle[1], T(2π / l.npoints))
    x = ps.radius[1] .* cos.(angles) .+ ps.center[1]
    y = ps.radius[1] .* sin.(angles) .+ ps.center[2]
    z = ps.center[3] .* ones(T, length(angles))
    return hcat(x, y, z), st
end
Lux.initialparameters(rng::AbstractRNG, ::NPCLayer{T}) where {T} = (center=rand(rng, T, 1, 3), offset_angle=rand(rng, T, 1, 1) .* T(π / 4), radius=[T(2)],)
Lux.initialstates(::AbstractRNG, ::NPCLayer) = NamedTuple()
Lux.parameterlength(l::NPCLayer) = 5
Lux.statelength(::NPCLayer) = 0
