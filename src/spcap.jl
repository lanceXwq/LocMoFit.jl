"""
    SphericalCap()

A spherical cap of constant area 4πs², with its apex at its position `(x, y, z)` and its outward normal along its axis. Its shape is fitted as `alpha`, the half-angle from the axis to the rim seen from the sphere's center, which gives the dimensionless curvature κ′ = sin(`alpha`/2) = curvature × `s` (derivations/derivations.qmd, @eq-cap-alpha): 0 is a flat disc of radius 2s, ±π/2 a half-sphere, ±π the whole sphere of radius s, and a negative `alpha` bends the other way. The sphere's radius is s/κ′. This is DynMoSim's `Bend`, and the cap MATLAB LocMoFit parametrizes by its surface area and close angle.
"""
struct SphericalCap <: SimpleModel end

function shape_parameters(::SphericalCap, dtype::Type{<:AbstractFloat})
    return (
        s=convert(dtype, 50), # area 4π·50² nm²
        alpha=convert(dtype, 0.961), # a half-angle of 55°, κ′ ≈ 0.46
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
    κ = sin(shape.alpha / 2)
    h = view(unit_cap, 3:3, :)
    # one fused broadcast over the 3×M points (rows 1–2 the distance from the axis times ρ̂, row 3 the depth), so that only the result is allocated: this halves what a gradient allocates
    return shape.s .* ifelse.((1:3) .== 3, -κ .* h, sqrt.(h .* (2 .- κ^2 .* h)) .* unit_cap)
end

# the scale, with the default prior on sizes, and the curvature κ′ (`kappa`, fitted as the half-angle `alpha`), uniform on [-1, 1] by default
function shape_quantities(::SphericalCap, suffix)
    s = Symbol(:s, suffix)
    return (
        s => (Size{s}(), SIZE_PRIOR),
        Symbol(:kappa, suffix) => (Curvature{Symbol(:alpha, suffix),s}(), Flat()),
    )
end

# the cap's size and shape in the usual terms
function shape_summary(::SphericalCap, shape)
    κ = sin(shape.alpha / 2)
    return (
        s=shape.s,
        kappa=κ,
        curvature=κ / shape.s,
        radius=shape.s / κ,
        alpha=acos(1 - 2κ^2),
        area=4π * shape.s^2,
    )
end
