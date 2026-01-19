unit_ring(T::Type{<:AbstractFloat}, N::Integer) = unit_ring(map(T, range(0, 2π * (1 - 1 / N), length=N)))

function unit_ring(φ::AbstractVector)
    X = similar(φ, 3, length(φ))
    unit_ring!(X, φ)
    return X
end

function unit_ring!(X::AbstractMatrix{T}, φ::AbstractVector{T}) where {T}
    X[1, :] .= cos.(φ)
    X[2, :] .= sin.(φ)
    X[3, :] .= 0
    return X
end

# function circle_points!(X::AbstractMatrix{T}, φ::AbstractVector{T}, r::T, φ₀::T) where {T}
#     α = φ .+ φ₀
#     X[1, :] .= r .* cos.(α)
#     X[2, :] .= r .* sin.(α)
#     X[3, :] .= 0
#     return X
# end

const GOLDEN_ANGLE = (3 - √5)π

function unit_sphere(T::Type{<:AbstractFloat}=Float32, N::Integer=100; γ::Real=GOLDEN_ANGLE, mode::Symbol=:Fibonacci)
    X = Matrix{T}(undef, 3, N)
    unit_sphere!(X; γ=γ, mode=mode)
    return X
end

unit_sphere!(X::AbstractMatrix{T}; γ::Real=GOLDEN_ANGLE, mode::Symbol=:Fibonacci) where {T} = unit_sphere!(X, convert(T, γ), Val(mode))
function unit_sphere!(X::AbstractMatrix{T}, γ::T, ::Val{:Fibonacci}) where {T}
    N = size(X, 2)
    i = 0:N-1
    z = 1 .- (i .+ T(0.5)) .* T(2 / N)
    r = sqrt.(1 .- z .^ 2)
    φ = i .* γ
    X[1, :] .= r .* cos.(φ)
    X[2, :] .= r .* sin.(φ)
    X[3, :] .= z
    return X
end

function unit_disk(T::Type{<:AbstractFloat}=Float32, N::Integer=100; γ::Real=GOLDEN_ANGLE, mode::Symbol=:Fibonacci)
    X = Matrix{T}(undef, 3, N)
    unit_disk!(X; γ=γ, mode=mode)
    return X
end

unit_disk!(X::AbstractMatrix{T}; γ::Real=GOLDEN_ANGLE, mode::Symbol=:Fibonacci) where {T} = unit_disk!(X, convert(T, γ), Val(mode))
function unit_disk!(X::AbstractMatrix{T}, γ::T, ::Val{:Fibonacci}) where {T}
    N = size(X, 2)
    i = 0:N-1
    r = sqrt.((i .+ T(0.5)) ./ N)
    φ = i .* γ
    X[1, :] .= r .* cos.(φ)
    X[2, :] .= r .* sin.(φ)
    X[3, :] .= 0
    return X
end
