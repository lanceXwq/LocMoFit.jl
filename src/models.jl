struct NPCLayer{T} <: Lux.AbstractLuxLayer
    half_precision::T # precision here is the reciprocal of variance
    log_const::T
    NPCLayer(stddev::T) where {T} = new{T}(stddev^-2 / 2, -log(stddev^2 * 2π) / 2)
end

function (l::NPCLayer{T})(_, ps, st) where {T}
    angles = T.([0.0
        0.7853981633974483
        1.5707963267948966
        2.356194490192345
        3.141592653589793
        3.9269908169872414
        4.71238898038469
        5.497787143782138]) .+ mod(ps.offset_angle[1], T(π / 4))
    x = ps.radius[1] .* cos.(angles) .+ ps.center[1]
    y = ps.radius[1] .* sin.(angles) .+ ps.center[2]
    z = fill!(similar(x), ps.center[3])
    return hcat(x, y, z), st
end
Lux.initialparameters(rng::AbstractRNG, ::NPCLayer{T}) where {T} = (center=rand(rng, T, 1, 3), offset_angle=rand(rng, T, 1, 1) .* T(π / 4), radius=[T(2)],)
Lux.initialstates(::AbstractRNG, ::NPCLayer) = NamedTuple()
Lux.parameterlength(l::NPCLayer) = 5
Lux.statelength(::NPCLayer) = 0
