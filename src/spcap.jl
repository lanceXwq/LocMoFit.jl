struct SphericalCap <: Model end

initialparameters(::SphericalCap, dtype::Type{<:AbstractFloat}) = (
    # apex=zeros(dtype, 3),
    x=convert(dtype, 0.0),
    y=convert(dtype, 0.0),
    z=convert(dtype, 0.0),
    c=convert(dtype, 0.02), # 50 nm radius
    α=convert(dtype, π / 2),
    θ=convert(dtype, 0.0),
    ϕ=convert(dtype, 0.0),
    σ=convert(dtype, 0.0),
)

function unit_model(::SphericalCap, ps; spacing::Real)
    area = 4π / ps.c^2
    npoints = ceil(Int, area / spacing^2)
    return unit_sphere(eltype(ps[1]), npoints)
end

LuxCore.initialstates(::AbstractRNG, ::SphericalCap) = NamedTuple()
LuxCore.parameterlength(::SphericalCap) = 8
LuxCore.statelength(::SphericalCap) = 0

function (::SphericalCap)(unit_sphere, ps, st)
    fullps = fullparams(ps, st)
    X = unit_sphere_to_cap(unit_sphere, fullps.α)
    r = 1 / fullps.c
    scale!(X, r)
    X[3, :] .-= r
    rot_trans!(X, fullps.θ, fullps.ϕ, fullps.x, fullps.y, fullps.z)
    # rot_trans!(X, fullps.θ, fullps.ϕ, fullps.apex)
    return X, st
end