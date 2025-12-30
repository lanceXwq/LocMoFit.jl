struct NPCLayer{T} <: Lux.AbstractLuxLayer
    npoints::Int
    half_precision::T # precision here is the reciprocal of variance
    log_const::T
    NPCLayer(stddev::T, npoints=8) where {T} = new{T}(npoints, stddev^-2 / 2, -log(stddev^2 * 2π) / 2)
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

struct NPC3DLayer{T} <: Lux.AbstractLuxLayer
    npoints::Int
    half_precision::T # precision here is the reciprocal of variance
    log_const::T
    NPC3DLayer(stddev::T, npoints=8) where {T} = new{T}(npoints, stddev^-2 / 2, -log(stddev^2 * 2π) / 2)
end

function (l::NPC3DLayer{T})(_, ps, st) where {T}
    angles = T.([2π / l.npoints * n for n in 0:(l.npoints-1)]) .+ mod(ps.offset_angle[1], T(2π / l.npoints))
    x = ps.radius[1] .* cos.(angles) .+ ps.center[1]
    y = ps.radius[1] .* sin.(angles) .+ ps.center[2]
    z = ps.center[3] .* ones(T, length(angles))
    x2 = ps.radius[1] .* cos.(angles) .+ ps.center[1]
    y2 = ps.radius[1] .* sin.(angles) .+ ps.center[2]
    z2 = (ps.center[3] + ps.distance[1]) .* ones(T, length(angles))
    return hcat(vcat(x, x2), vcat(y, y2), vcat(z, z2)), st
end

Lux.initialparameters(rng::AbstractRNG, ::NPC3DLayer{T}) where {T} = (center=rand(rng, T, 1, 3), offset_angle=rand(rng, T, 1, 1) .* T(π / 4), radius=[T(2)], distance=[T(5)],)
Lux.initialstates(::AbstractRNG, ::NPC3DLayer) = NamedTuple()
Lux.parameterlength(l::NPC3DLayer) = 6
Lux.statelength(::NPC3DLayer) = 0