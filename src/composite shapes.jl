# abstract type CompositeLayer{layers} <: LuxCore.AbstractLuxContainerLayer{layers} end

const NPC_PARAM_SYMS = (:x, :y, :z, :r, :φ₀, :θ, :ϕ, :Δx, :Δy, :Δz, :Δr, :Δφ₀, :Δθ, :Δϕ)

# struct NuclearPoreComplex{R1,R2} <: LuxCore.AbstractLuxContainerLayer{(:ring1, :ring2)}
#     ring1::R1
#     ring2::R2
# end

struct NuclearPoreComplex{Fixed} <: LuxCore.AbstractLuxLayer end
NuclearPoreComplex(; fixed::Tuple{Vararg{Symbol}}=(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)) = NuclearPoreComplex{fixed}()

LuxCore.initialparameters(rng::AbstractRNG, ::NuclearPoreComplex{(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)}) = ComponentArray(
    x=zero(Float32),
    y=zero(Float32),
    z=zero(Float32),
    r=one(Float32),
    φ₀=zero(Float32),
    θ=one(Float32),
    ϕ=one(Float32),
    Δz=one(Float32),
) # x * 2π is not equivalent to x * 2 * π
LuxCore.initialstates(::AbstractRNG, ::NuclearPoreComplex{(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)}) = NamedTuple()
LuxCore.parameterlength(::NuclearPoreComplex{(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)}) = 8
LuxCore.statelength(::NuclearPoreComplex{(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)}) = 0

# LuxCore.initialparameters(rng::AbstractRNG, ::Ring) = (
#     x=[_rand(rng, scale=2, shift=-1)],
#     y=[_rand(rng, scale=2, shift=-1)],
#     z=[_rand(rng, scale=2, shift=-1)],
#     r=[_rand(rng, scale=10)],
#     φ₀=zeros(Float32, 1),
#     θ=[_rand(rng, scale=π)],
#     ϕ=[_rand(rng, scale=2π)]) # x * 2π is not equivalent to x * 2 * π
# LuxCore.initialstates(::AbstractRNG, ::Ring) = NamedTuple()
# LuxCore.parameterlength(l::Ring) = 7
# LuxCore.statelength(::Ring) = 0

function (npc::NuclearPoreComplex{(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)})(φ₁::AbstractVector{T}, φ₂::AbstractVector{T}, ps, st) where {T}
    N₁, N₂ = length(φ₁), length(φ₂)
    X = similar(φ₁, 3, N₁ + N₂)
    circle_points!(view(X, :, 1:N₁), φ₁, ps.r, ps.φ₀)
    circle_points!(view(X, :, N₁+1:N₁+N₂), φ₂, ps.r, ps.φ₀)
    trans!(view(X, :, N₁+1:N₁+N₂), ps.Δz, 3)
    rot_trans!(X, ps.θ, ps.ϕ, view(ps, 1:3))
    return X, st
end
(npc::NuclearPoreComplex{(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)})(φ::AbstractVector{T}, ps, st) where {T} = (npc::NuclearPoreComplex{(:Δx, :Δy, :Δr, :Δφ₀, :Δθ, :Δϕ)})(φ, φ, ps, st)

# abstract type CompositeLayer{layers} <: LuxCore.AbstractLuxContainerLayer{layers} end

# const NPC_PARAM_SYMS = (:x, :y, :z, :r, :φ₀, :θ, :ϕ)

# struct NuclearPoreComplex{R1<:Ring,R2<:Ring,Fixed} <: CompositeLayer{(:ring1, :ring2)}
#     ring1::R1
#     ring2::R2
# end

# NuclearPoreComplex(; fixed::Tuple{Vararg{Symbol}}=(:x, :y, :r, :φ₀, :θ, :ϕ)) = NuclearPoreComplex{Ring,Ring,fixed}(Ring(), Ring())

# LuxCore.initialparameters(rng::AbstractRNG, ::Ring) = (
#     x=[_rand(rng, scale=2, shift=-1)],
#     y=[_rand(rng, scale=2, shift=-1)],
#     z=[_rand(rng, scale=2, shift=-1)],
#     r=[_rand(rng, scale=10)],
#     φ₀=zeros(Float32, 1),
#     θ=[_rand(rng, scale=π)],
#     ϕ=[_rand(rng, scale=2π)]) # x * 2π is not equivalent to x * 2 * π
# LuxCore.initialstates(::AbstractRNG, ::Ring) = NamedTuple()
# LuxCore.parameterlength(l::Ring) = 7
# LuxCore.statelength(::Ring) = 0

# function (npc::NuclearPoreComplex{Ring,Ring,(:x, :y, :r, :φ₀, :θ, :ϕ)})(φ::AbstractVector{T}, ps, st) where {T}
#     N = length(φ)
#     X = Matrix{T}(undef, 3, 2 * N)
#     ring_points!(view(X, :, 1:N), φ, ps.r[], ps.φ₀[])
#     ring_points!(view(X, :, N+1:2N), φ, ps.r[], ps.φ₀[])
#     rot_trans!(view(X, :, N+1:2N), zero(T), zero(T), [zero(T), zero(T), ps.z2[]])
#     rot_trans!(X, ps.θ[], ps.ϕ[], [ps.x[], ps.y[], ps.z[]])
#     return X, st
# end

# function LuxCore.setup(rng::AbstractRNG, npc::NuclearPoreComplex{<:Ring{N1,V1},<:Ring{N2,V2}}) where {T,N1,N2,V1<:AbstractVector{T},V2<:AbstractVector{T}}
#     ps1 = LuxCore.initialparameters(rng, npc.ring1)
#     # (pre_rot_x, pre_rot_y, pre_rot_z, pre_rot_φ, r, θ, ϕ) = [c ? v : rand(rng, T, size(v)) .* s for (c, v, s) in zip(npc.coupled, ps1, T.((1.0, 1.0, 1.0, 2π / N2, 1.0, π, 2π)))]
#     ps2 = (x=zeros(T, 1), y=zeros(T, 1), z=ones(T, 1), φ₀=zeros(T, 1), r=zeros(T, 1), θ=zeros(T, 1), ϕ=zeros(T, 1))
#     return (ring1=ps1, ring2=ps2), LuxCore.initialstates(rng, npc)
# end

# @generated function _ring_ps(ps, ::Val{SHARED}, ::Val{WHICH}) where {SHARED,WHICH}
#     all = RING_PARAM_SYMS
#     shared = SHARED
#     ring_field = WHICH === 1 ? :ring1 : :ring2

#     keys_expr = Expr(:tuple, (QuoteNode(s) for s in all)...)

#     vals = Vector{Expr}(undef, length(all))
#     for (i, s) in enumerate(all)
#         if s in shared
#             vals[i] = :(ps.shared.$s)
#         else
#             vals[i] = :(ps.$ring_field.$s)
#         end
#     end
#     vals_expr = Expr(:tuple, vals...)

#     return :(NamedTuple{$keys_expr}($vals_expr))
# end

# function (m::NuclearPoreComplex{R1,R2,SHARED})(input, ps, st) where {R1,R2,SHARED}
#     ps1 = _ring_ps(ps, Val(SHARED), Val(1))
#     ps2 = _ring_ps(ps, Val(SHARED), Val(2))

#     X1, st1 = m.ring1(input, ps1, st.ring1)
#     X2, st2 = m.ring2(input, ps2, st.ring2)

#     return vcat(X1, X2), (ring1=st1, ring2=st2)
# end