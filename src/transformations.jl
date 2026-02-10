# const VecOrTupleAnyN{T} = Union{AbstractVector{T},Tuple{Vararg{T}}}

trans!(X::AbstractMatrix{T}, Δ𝐱::AbstractVector{T}) where {T} = X .+= Δ𝐱
function trans!(X::AbstractMatrix{T}, Δ::T; dims::Integer) where {T}
    X[dims, :] .+= Δ
    return X
end

scale!(X::AbstractMatrix{T}, s::T) where {T} = lmul!(s, X)

scale!(X::AbstractMatrix{T}, s::AbstractVector{T}) where {T} = X .*= s

function rotmat(θ::T, ϕ::T) where {T}
    sθ, cθ = sincos(θ)
    sϕ, cϕ = sincos(ϕ)
    t = one(T) - cθ
    return [
        one(T)-t*cϕ*cϕ -t*sϕ*cϕ sθ*cϕ;
        -t*sϕ*cϕ one(T)-t*sϕ*sϕ sθ*sϕ;
        -sθ*cϕ -sθ*sϕ cθ
    ]
end

function rotmat(θ::AbstractVector{T}, ϕ::AbstractVector{T}) where {T}
    sθ, cθ = sin.(θ), cos.(θ)
    sϕ, cϕ = sin.(ϕ), cos.(ϕ)
    t = one(T) .- cθ

    a11 = one(T) .- t .* cϕ .* cϕ
    a12 = .-t .* sϕ .* cϕ
    a13 = sθ .* cϕ

    a21 = .-t .* sϕ .* cϕ
    a22 = one(T) .- t .* sϕ .* sϕ
    a23 = sθ .* sϕ

    a31 = .-sθ .* cϕ
    a32 = .-sθ .* sϕ
    a33 = cθ

    # 3×3 array on the same device as θ/ϕ
    return vcat(
        hcat(a11, a12, a13),
        hcat(a21, a22, a23),
        hcat(a31, a32, a33),
    )
end

# function rodrigues(ax, ay, az, theta, eps: float = 1e-10):
#     # Normalize axis
#     norm = torch.sqrt(ax**2 + ay**2 + az**2).clamp_min(eps)
#     x = ax / norm
#     y = ay / norm
#     z = az / norm

#     c = torch.cos(theta)
#     s = torch.sin(theta)
#     C = 1 - c

#     return torch.stack(
#         [
#             torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], dim=-1),
#             torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], dim=-1),
#             torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], dim=-1),
#         ],
#         dim=-2,
#     ).squeeze_(0)
# end

# rotmat(θ::T, ϕ::T) where {T} = rotmat(θ, ϕ)
# isapprox(θ, 0, atol=1e-10) ? RotZ(ϕ) : rodrigues(θ, ϕ)
# AngleAxis(θ, -sin(ϕ), cos(ϕ), 0)

# Slower but it takes vector input, making GPU execution easier. Also more robust (works even if rotvec == [0, 0])
rotmat(rotvec::AbstractVector{T}) where {T} = RotationVec(rotvec..., zero(T))

rot_trans!(X::AbstractMatrix{T}, θ::T, ϕ::T, x::T, y::T, z::T) where {T} = X .= fused_dense_bias_activation(identity, rotmat(θ, ϕ), X, [x, y, z])
rot_trans!(X::AbstractMatrix{T}, θ::T, ϕ::T, Δ𝐱::AbstractVector{T}) where {T} = X .= fused_dense_bias_activation(identity, rotmat(θ, ϕ), X, Δ𝐱)
rot_trans!(X::AbstractMatrix{T}, θ::AbstractVector{T}, ϕ::AbstractVector{T}, Δ𝐱::AbstractVector{T}) where {T} = X .= fused_dense_bias_activation(identity, rotmat(θ, ϕ), X, Δ𝐱)
# if isapprox(θ, 0, atol=1e-10) && isapprox(ϕ, 0, atol=1e-10)
#     trans!(X, Δ𝐱)
# else
#     X .= muladd(rotmat(θ, ϕ), X, Δ𝐱)
# end

function unit_sphere_to_cap(sphere_pts::AbstractMatrix{T}, half_angle::T) where {T}
    x = view(sphere_pts, 1, :)
    y = view(sphere_pts, 2, :)
    z = view(sphere_pts, 3, :)

    ca = cos(half_angle)
    a = (1 + ca) / 2
    b = (1 - ca) / 2

    cap_pts = similar(sphere_pts)
    cap_pts[3, :] .= muladd.(b, z, a)

    r1 = sqrt.(x .^ 2 .+ y .^ 2)
    r2 = sqrt.(1 .- view(cap_pts, 3, :) .^ 2)
    s = r2 ./ r1

    cap_pts[1, :] .= x .* s
    cap_pts[2, :] .= y .* s
    return cap_pts
end

function unit_sphere_to_cap(sphere_pts::AbstractMatrix{T}, half_angle::AbstractVector{T}) where {T}
    x = view(sphere_pts, 1, :)
    y = view(sphere_pts, 2, :)
    z = view(sphere_pts, 3, :)

    ca = @. cos(half_angle)
    a = @. (1 + ca) / 2
    b = @. (1 - ca) / 2

    cap_pts = similar(sphere_pts)
    cap_pts[3, :] .= muladd.(b, z, a)

    r1 = sqrt.(x .^ 2 .+ y .^ 2)
    r2 = sqrt.(1 .- view(cap_pts, 3, :) .^ 2)
    s = r2 ./ r1
    
    cap_pts[1, :] .= x .* s
    cap_pts[2, :] .= y .* s
    return cap_pts
end

function unit_ring_to_arc(ring_pts::AbstractMatrix{T}, half_angle::T) where {T}
    iseven(size(ring_pts, 2)) || (@warn "Even number of points for unit_ring_to_arc is risky" maxlog = 2)
    φ = asin.(view(ring_pts, 1, :)) .* (half_angle / π)
    arc_pts = similar(ring_pts)
    arc_pts[1, :] .= sin.(φ)
    arc_pts[2, :] .= cos.(φ)
    arc_pts[3, :] .= 0
    return arc_pts
end