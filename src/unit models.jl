# N equally spaced points on the unit circle in the xy-plane, as a 3×N matrix
function unit_ring(T::Type{<:AbstractFloat}, N::Integer)
    φ = map(T, range(0, 2π * (1 - 1 / N); length=N))
    X = similar(φ, 3, N)
    X[1, :] .= cos.(φ)
    X[2, :] .= sin.(φ)
    X[3, :] .= 0
    return X
end

const GOLDEN_ANGLE = (3 - √5)π

# N equal-area points on the unit sphere (a Fibonacci lattice), as a 3×N matrix
function unit_sphere(T::Type{<:AbstractFloat}, N::Integer)
    X = Matrix{T}(undef, 3, N)
    i = 0:(N - 1)
    z = 1 .- (i .+ T(0.5)) .* T(2 / N)
    r = sqrt.(1 .- z .^ 2)
    φ = i .* convert(T, GOLDEN_ANGLE)
    X[1, :] .= r .* cos.(φ)
    X[2, :] .= r .* sin.(φ)
    X[3, :] .= z
    return X
end

# The unit sphere's points as (ρ̂₁, ρ̂₂, h): the direction of each point around the axis and its depth below the north pole, h = 1 - z ∈ (0, 2), which the constant-area cap needs (SphericalCap).
function unit_cap(T::Type{<:AbstractFloat}, N::Integer)
    X = unit_sphere(T, N)
    ρ = sqrt.(view(X, 1, :) .^ 2 .+ view(X, 2, :) .^ 2)
    return permutedims(hcat(view(X, 1, :) ./ ρ, view(X, 2, :) ./ ρ, 1 .- view(X, 3, :)))
end
