trans!(X::AbstractMatrix{T}, Δ𝐱::AbstractVector{T}) where {T} = X .+= Δ𝐱
function trans!(X::AbstractMatrix{T}, Δ::T, dims::Integer) where {T}
    X[dims, :] .+= Δ
    return X
end

rotmat(θ::T, ϕ::T) where {T} =
    isapprox(θ, 0, atol=1e-10) ? RotZ(ϕ) : AngleAxis(θ, -sin(θ) * sin(ϕ), sin(θ) * cos(ϕ), 0)

rot_trans!(X::AbstractMatrix{T}, θ::T, ϕ::T, Δ𝐱::AbstractVector{T}) where {T} =
    if isapprox(θ, 0, atol=1e-10) && isapprox(ϕ, 0, atol=1e-10)
        trans!(X, Δ𝐱)
    else
        X .= muladd(rotmat(θ, ϕ), X, Δ𝐱)
    end