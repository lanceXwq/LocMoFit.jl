"""
    SphericalCap()

A spherical cap of constant area 4πs², with its apex at its position `(x, y, z)` and its outward normal along its axis. Its dimensionless curvature is κ′ = tanh(`eta`) = curvature × `s`: 0 is a flat disc of radius 2s, ±1 the whole sphere of radius s, and a negative κ′ bends the other way. The sphere's radius is s/κ′ and the half-angle from the axis to the rim, seen from the sphere's center, is acos(1 − 2κ′²) (derivations/derivation.qmd). This is SimSMLM's `Bend`, and the cap MATLAB LocMoFit parametrizes by its surface area and close angle.
"""
struct SphericalCap <: SimpleModel end

function shape_parameters(::SphericalCap, dtype::Type{<:AbstractFloat})
    return (
        s=convert(dtype, 50), # area 4π·50² nm²
        eta=convert(dtype, 0.5), # κ′ = tanh(0.5) ≈ 0.46
    )
end

# equal-area points on the unit sphere, as many as the cap's area holds at the spacing
function unit_points(::SphericalCap, shape; spacing::Real)
    area = 4π * shape.s^2
    npoints = ceil(Int, area / spacing^2)
    return unit_cap(typeof(shape.s), npoints)
end

# the unit sphere's point at depth h below its north pole goes to depth s·κ′·h below the apex, at distance s·√(h(2 − κ′²h)) from the axis; each point keeps the area above it, so equal-area points stay equal-area
function local_points(::SphericalCap, unit_cap, shape)
    κ = tanh(shape.eta)
    h = view(unit_cap, 3:3, :)
    # one fused broadcast over the 3×M points (rows 1–2 the distance from the axis times ρ̂, row 3 the depth), so that only the result is allocated: this halves what a gradient allocates
    return shape.s .* ifelse.((1:3) .== 3, -κ .* h, sqrt.(h .* (2 .- κ^2 .* h)) .* unit_cap)
end

shape_penalty(::SphericalCap, shape) = sq_hinge(1 / shape.s^2 - 1)

# the cap's size and shape in the usual terms
function shape_summary(::SphericalCap, shape)
    κ = tanh(shape.eta)
    return (
        s=shape.s,
        kappa=κ,
        curvature=κ / shape.s,
        radius=shape.s / κ,
        alpha=acos(1 - 2κ^2),
        area=4π * shape.s^2,
    )
end
