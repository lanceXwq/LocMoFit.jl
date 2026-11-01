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

"""
    chart_rotation(R₀, vx, vy)

The rotation that orients a model whose axis has tangent-chart coordinates `(vx, vy)`. The chart's frame `R₀` (from `rotmat(θ₀, ϕ₀)` of the starting axis) turns z to its center; the axis is `R₀·m` with `m = normalize((vx, vy, 1))`, and the rotation is `R₀` times the minimal rotation taking z to `m`. `m₃ > 0` for every `(vx, vy)`, so that minimal rotation never meets its singularity (at `m = -z`), and the chart reaches every axis within 90° of its center (derivations/derivation.qmd).
"""
function chart_rotation(R₀::AbstractMatrix, vx, vy)
    r = inv(sqrt(1 + vx^2 + vy^2))
    m₁, m₂, m₃ = vx * r, vy * r, r
    k = inv(1 + m₃)
    M = [
        1-k*m₁^2 -k*m₁*m₂ m₁
        -k*m₁*m₂ 1-k*m₂^2 m₂
        -m₁ -m₂ m₃
    ]
    return R₀ * M
end

# rotate the points X (3×M) by R, then shift them by (x, y, z); returns new points
place(X::AbstractMatrix, R::AbstractMatrix, x, y, z) = R * X .+ [x, y, z]
