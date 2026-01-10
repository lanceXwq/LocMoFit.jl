const NPC_PARAM_SYMS = (:x, :y, :z, :r, :φ₀, :θ, :ϕ, :Δx, :Δy, :Δz, :Δr, :Δφ₀, :Δθ, :Δϕ)

struct NuclearPoreComplex{T,Fixed} <: LuxCore.AbstractLuxLayer end
NuclearPoreComplex(T::Type{<:AbstractFloat}=Float32; fixed::Tuple{Vararg{Symbol}}=(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)) = NuclearPoreComplex{T,(sort(collect(fixed))...,)}()

# Shared functions
LuxCore.initialstates(::AbstractRNG, ::NuclearPoreComplex) = NamedTuple()
LuxCore.statelength(::NuclearPoreComplex) = 0
(npc::NuclearPoreComplex)(φ, ps, st) = (npc)(φ, φ, ps, st)

# Function definitions for NuclearPoreComplex{T,(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)}
LuxCore.initialparameters(rng::AbstractRNG, ::NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δθ, :Δφ₀, :Δϕ)}) where {T} = ComponentArray(
    x=randn(rng, T),
    y=randn(rng, T),
    z=randn(rng, T),
    r=rand(rng, T),
    φ₀=rand(rng, T),
    θ=rand(rng, T),
    ϕ=rand(rng, T),
    Δz=randn(rng, T),
)
LuxCore.parameterlength(::NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δθ, :Δφ₀, :Δϕ)}) where {T} = 8

function (npc::NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δθ, :Δφ₀, :Δϕ)})(φ₁::AbstractVector{T}, φ₂::AbstractVector{T}, ps, st) where {T}
    N₁, N₂ = length(φ₁), length(φ₂)
    X = similar(φ₁, 3, N₁ + N₂)
    circle_points!(view(X, :, 1:N₁), φ₁, ps.r, ps.φ₀)
    circle_points!(view(X, :, N₁+1:N₁+N₂), φ₂, ps.r, ps.φ₀)
    trans!(view(X, :, N₁+1:N₁+N₂), ps.Δz, 3)
    rot_trans!(X, ps.θ, ps.ϕ, view(ps, 1:3))
    return X, st
end

# Function definitions for NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δφ₀)}
LuxCore.initialparameters(rng::AbstractRNG, ::NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δφ₀)}) where {T} = ComponentArray(
    x=randn(rng, T),
    y=randn(rng, T),
    z=randn(rng, T),
    r=rand(rng, T),
    φ₀=rand(rng, T),
    θ=rand(rng, T),
    ϕ=rand(rng, T),
    Δz=randn(rng, T),
    Δθ=rand(rng, T),
    Δϕ=rand(rng, T),
)
LuxCore.parameterlength(::NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δφ₀)}) where {T} = 10

function (npc::NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δφ₀)})(φ₁::AbstractVector{T}, φ₂::AbstractVector{T}, ps, st) where {T}
    N₁, N₂ = length(φ₁), length(φ₂)
    X = similar(φ₁, 3, N₁ + N₂)
    circle_points!(view(X, :, 1:N₁), φ₁, ps.r, ps.φ₀)
    circle_points!(view(X, :, N₁+1:N₁+N₂), φ₂, ps.r, ps.φ₀)
    rot_trans!(view(X, :, N₁+1:N₁+N₂), ps.Δθ, ps.Δϕ, [zero(T), zero(T), ps.Δz])
    rot_trans!(X, ps.θ, ps.ϕ, view(ps, 1:3))
    return X, st
end
