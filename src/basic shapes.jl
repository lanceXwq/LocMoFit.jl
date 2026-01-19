# _rand(rng::AbstractRNG, T::Type{<:Real}=Float32; scale=one(T), shift=zero(T)) = muladd(rand(rng, T), T(scale), T(shift))

const RING_SYMS = (:x, :y, :z, :r, :φ₀, :θ, :ϕ)
struct Ring{T,F} <: LuxCore.AbstractLuxLayer end
Ring(T::Type{<:AbstractFloat}=Float32; fixed::Tuple{Vararg{Symbol}}=()) = Ring{T,fixed}()

LuxCore.initialparameters(rng::AbstractRNG, ::Ring{T}) where {T} = ComponentArray(
    x=randn(rng, T),
    y=randn(rng, T),
    z=randn(rng, T),
    r=rand(rng, T),
    φ₀=rand(rng, T),
    θ=rand(rng, T),
    ϕ=rand(rng, T),
)
LuxCore.initialstates(::AbstractRNG, ::Ring) = NamedTuple()
LuxCore.parameterlength(::Ring) = 7
LuxCore.statelength(::Ring) = 0
setup(rng::AbstractRNG, r::Ring{T}; npoints::Integer=100) where {T} = LuxCore.setup(rng, r)..., unit_ring(T, npoints)

function (r::Ring)(unit_ring, ps, st)
    X = unit_ring .* ps.r
    rot_trans!(X, ps.θ, ps.ϕ, view(ps, 1:3))
    return X, st
end

const DISK_SYMS = (:x, :y, :z, :r, :φ₀, :θ, :ϕ)
struct Disk{T,F} <: LuxCore.AbstractLuxLayer end
Disk(T::Type{<:AbstractFloat}=Float32; fixed::Tuple{Vararg{Symbol}}=()) = Disk{T,fixed}()

LuxCore.initialparameters(rng::AbstractRNG, ::Disk{T}) where {T} = ComponentArray(
    x=randn(rng, T),
    y=randn(rng, T),
    z=randn(rng, T),
    r=rand(rng, T),
    φ₀=rand(rng, T),
    θ=rand(rng, T),
    ϕ=rand(rng, T),
)
LuxCore.initialstates(::AbstractRNG, ::Disk) = NamedTuple()
LuxCore.parameterlength(::Disk) = 7
LuxCore.statelength(::Disk) = 0
setup(rng::AbstractRNG, r::Disk{T}; npoints::Integer=100) where {T} = LuxCore.setup(rng, r)..., unit_disk(T, npoints)

function (r::Disk)(unit_model, ps, st)
    X = unit_model .* ps.r
    rot_trans!(X, ps.θ, ps.ϕ, view(ps, 1:3))
    return X, st
end

const SPCAP_SYMS = (:x, :y, :z, :r, :ϑ, :φ₀, :θ, :ϕ)
struct SphericalCap{T,F} <: LuxCore.AbstractLuxLayer end
SphericalCap(T::Type{<:AbstractFloat}=Float32; fixed::Tuple{Vararg{Symbol}}=()) = SphericalCap{T,fixed}()

LuxCore.initialparameters(rng::AbstractRNG, ::SphericalCap{T}) where {T} = ComponentArray(
    x=randn(rng, T),
    y=randn(rng, T),
    z=randn(rng, T),
    c=rand(rng, T),
    ϑ=rand(rng, T),
    φ₀=rand(rng, T),
    θ=rand(rng, T),
    ϕ=rand(rng, T),
)
LuxCore.initialstates(::AbstractRNG, ::SphericalCap) = NamedTuple()
LuxCore.parameterlength(::SphericalCap) = 8
LuxCore.statelength(::SphericalCap) = 0
setup(rng::AbstractRNG, sc::SphericalCap{T}; npoints::Integer=100) where {T} = LuxCore.setup(rng, sc)..., unit_sphere(T, npoints)

function (::SphericalCap{T})(unit_sphere, ps, st) where {T}
    X = unit_sphere_to_cap(unit_sphere, ps.ϑ)
    r = 1 / ps.c
    scale!(X, r)
    trans!(X, -r, dims=3)
    rot_trans!(X, ps.θ, ps.ϕ, view(ps, 1:3))
    return X, st
end