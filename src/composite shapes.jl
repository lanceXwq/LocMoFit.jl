const NPC_PARAM_SYMS = (:x, :y, :z, :r, :φ₀, :θ, :ϕ, :Δx, :Δy, :Δz, :Δr, :Δφ₀, :Δθ, :Δϕ)

struct NuclearPoreComplex{T,F} <: LuxCore.AbstractLuxLayer end

NuclearPoreComplex(T::Type{<:AbstractFloat}=Float32; fixed::Tuple{Vararg{Symbol}}=(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)) = NuclearPoreComplex{T,(sort(collect(fixed))...,)}()

# function NuclearPoreComplex(T::Type{<:AbstractFloat}=Float32; npoints::Tuple{Int,Int}=(100, 100), fixed::Tuple{Vararg{Symbol}}=(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ))
#     X = Matrix{T}(undef, 3, sum(npoints))
#     N1, N2 = npoints
#     φ1 = map(T, range(0, 2π * (1 - 1 / N1), length=N1))
#     φ2 = map(T, range(0, 2π * (1 - 1 / N2), length=N2))
#     @views unit_ring!(X[:, 1:N1], φ1)
#     @views unit_ring!(X[:, N1+1:N1+N2], φ2)
#     return NuclearPoreComplex{T,(sort(collect(fixed))...,),typeof(Xdev)}(npoints, Xdev)
# end

# Shared functions
LuxCore.initialstates(::AbstractRNG, npc::NuclearPoreComplex) = NamedTuple()

LuxCore.statelength(npc::NuclearPoreComplex) = 0

setup(rng::AbstractRNG, npc::NuclearPoreComplex{T}; npoints=(100, 100)) where {T} = LuxCore.setup(rng, npc)..., (unit_ring(T, npoints[1]), unit_ring(T, npoints[2]))

# (npc::NuclearPoreComplex)(φ, ps, st) = (npc)(φ, φ, ps, st)

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

function (npc::NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δθ, :Δφ₀, :Δϕ)})(unit_model, ps, st) where {T}
    Xs = ps.r .* unit_model
    trans!(Xs[2], ps.Δz, dims=3)
    for X in Xs
        rot_trans!(X, ps.θ, ps.ϕ, view(ps, 1:3))
    end
    return Xs, st
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

function (npc::NuclearPoreComplex{T,(:Δr, :Δx, :Δy, :Δφ₀)})(unit_model, ps, st) where {T}
    Xs = ps.r .* unit_model
    rot_trans!(Xs[2], ps.Δθ, ps.Δϕ, [zero(T), zero(T), ps.Δz])
    for X in Xs
        rot_trans!(X, ps.θ, ps.ϕ, view(ps, 1:3))
    end
    return Xs, st
end



# struct NuclearPoreComplex2{T,M,Fixed}
#     N₁::Int
#     N₂::Int
#     pts::M
# end
# function NuclearPoreComplex2(T::Type{<:AbstractFloat}=Float32, (N1, N2)::Tuple{Int,Int}=(100, 100); fixed::Tuple{Vararg{Symbol}}=(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ))
#     pts = Matrix{T}(undef, 3, N1 + N2)
#     φ1 = map(T, range(0, 2π * (1 - 1 / N1), length=N1))
#     φ2 = map(T, range(0, 2π * (1 - 1 / N2), length=N2))
#     @views unit_circle_points!(pts[:, 1:N1], φ1)
#     @views unit_circle_points!(pts[:, N1+1:N1+N2], φ2)
#     NuclearPoreComplex2{T,typeof(pts),(sort(collect(fixed))...,)}(N1, N2, pts)
# end

