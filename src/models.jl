struct NPCLayer{T} <: LuxCore.AbstractLuxLayer
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

LuxCore.initialparameters(rng::AbstractRNG, ::NPCLayer{T}) where {T} = (center=rand(rng, T, 1, 3), offset_angle=rand(rng, T, 1) .* T(π / 4), radius=[T(2)],)
LuxCore.initialstates(::AbstractRNG, ::NPCLayer) = NamedTuple()
LuxCore.parameterlength(l::NPCLayer) = 5
LuxCore.statelength(::NPCLayer) = 0

struct NPC3DLayer{T} <: LuxCore.AbstractLuxLayer
    npoints::Int
    half_precision::T # precision here is the reciprocal of variance
    log_const::T
    NPC3DLayer(stddev::T, npoints=8) where {T} = new{T}(npoints, stddev^-2 / 2, -log(stddev^2 * 2π) / 2)
end

function points_on_circle!(X::AbstractMatrix{T}, radius::T, npoints::Int) where {T}
    angles = map(T, range(0, stop=2π - 2π / npoints, length=npoints))
    X[:, 1] .= radius .* cos.(angles)
    X[:, 2] .= radius .* sin.(angles)
    return X
end

function points_on_circle(radius::T, npoints::Int) where {T}
    X = Matrix{T}(undef, npoints, 3)
    points_on_circle!(X, radius, npoints)
end

function (l::NPC3DLayer{T})(_, ps, st) where {T}
    X = similar(ps.center, T, l.npoints * 2, length(ps.center))
    @views points_on_circle!(X[1:l.npoints, :], ps.radius[1], l.npoints)
    X[1:l.npoints, 3] .= -ps.distance[1] / 2
    @views copyto!(X[l.npoints+1:end, :], X[1:l.npoints, :])
    X[l.npoints+1:end, 3] .= ps.distance[1] / 2
    X .= X * rotmat(ps.orientation...)'
    X .+= ps.center'
    return X, st
end

LuxCore.initialparameters(rng::AbstractRNG, ::NPC3DLayer{T}) where {T} = (center=rand(rng, T, 3), offset_angle=rand(rng, T, 1) .* T(π / 4), radius=[T(2)], distance=[T(5)], orientation=rand(rng, T, 2),)
LuxCore.initialstates(::AbstractRNG, ::NPC3DLayer) = NamedTuple()
LuxCore.parameterlength(l::NPC3DLayer) = 8
LuxCore.statelength(::NPC3DLayer) = 0