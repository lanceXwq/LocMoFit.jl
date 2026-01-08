# abstract type ReferenceFrame end
# abstract type Absolute <: ReferenceFrame end
# abstract type Relative <: ReferenceFrame end

const RING_PARAM_SYMS = (:x, :y, :z, :r, :φ₀, :θ, :ϕ)

# Default initialization rules (customize as you like)
_rand(rng::AbstractRNG, T::Type{<:Real}=Float32; scale=one(T), shift=zero(T)) = muladd(rand(rng, T), T(scale), T(shift))
# init_param(rng::AbstractRNG, T::Type{<:Real}, ::Val{:x}) = [_rand(rng, T, scale=2, shift=-1)]
# init_param(rng::AbstractRNG, T::Type{<:Real}, ::Val{:y}) = [_rand(rng, T, scale=2, shift=-1)]
# init_param(rng::AbstractRNG, T::Type{<:Real}, ::Val{:z}) = [_rand(rng, T, scale=2, shift=-1)]
# init_param(rng::AbstractRNG, T::Type{<:Real}, ::Val{:r}) = [_rand(rng, T, scale=10)]
# init_param(rng::AbstractRNG, T::Type{<:Real}, ::Val{:φ₀}) = [_rand(rng, T, scale=2π)]
# init_param(rng::AbstractRNG, T::Type{<:Real}, ::Val{:θ}) = [_rand(rng, T, scale=π)]
# init_param(rng::AbstractRNG, T::Type{<:Real}, ::Val{:ϕ}) = [_rand(rng, T, scale=2π)]

struct Ring{Fixed} <: LuxCore.AbstractLuxLayer end
Ring(; fixed::Tuple{Vararg{Symbol}}=()) = Ring{fixed}()

# Ring(φ::V) where {V<:AbstractVector{<:AbstractFloat}} = Ring{eltype(φ),length(φ),V}(φ)
# function Ring(T::Type{<:AbstractFloat}=Float32, N::Integer=100)
#     φ = map(T, range(0, stop=2π - 2π / N, length=N))
#     Ring{N,typeof(φ)}(φ)
# end

LuxCore.initialparameters(rng::AbstractRNG, ::Ring) = ComponentArray(
    x=zero(Float32),
    y=zero(Float32),
    z=zero(Float32),
    r=one(Float32),
    φ₀=zero(Float32),
    θ=zero(Float32),
    ϕ=zero(Float32)) # x * 2π is not equivalent to x * 2 * π
LuxCore.initialstates(::AbstractRNG, ::Ring) = NamedTuple()
LuxCore.parameterlength(::Ring) = 7
LuxCore.statelength(::Ring) = 0

function circle_points!(X::AbstractMatrix{T}, φ::AbstractVector{T}, r::T, φ₀::T) where {T}
    α = φ .+ φ₀
    X[1, :] .= r .* cos.(α)
    X[2, :] .= r .* sin.(α)
    X[3, :] .= 0
    return X
end

function (r::Ring)(φ::AbstractVector{T}, ps, st) where {T}
    X = similar(φ, 3, length(φ))
    circle_points!(X, φ, ps.r[], ps.φ₀[])
    rot_trans!(X, ps.θ[], ps.ϕ[], view(ps, 1:3))
    return X, st
end

# function ring_effective_ps(ps_ring, st_ring)
#     fp = getfield(st_ring, :frozen_params)  # exists for frozen layer; for unfrozen you can set fp = NamedTuple()
#     # We need a way to read from ps if present, else from fp.
#     # For NamedTuples, `hasproperty` is available.
#     getv(sym) = hasproperty(ps_ring, sym) ? getfield(ps_ring, sym) : getfield(fp, sym)

#     return (x=getv(:x), y=getv(:y), z=getv(:z), r=getv(:r),
#         φ₀=getv(:φ₀), θ=getv(:θ), ϕ=getv(:ϕ))
# end