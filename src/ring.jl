"""
    Ring()

A circle of radius `r` (nm), centered at its position `(x, y, z)`, in the plane normal to its axis.
"""
struct Ring <: SimpleModel end

shape_parameters(::Ring, dtype::Type{<:AbstractFloat}) = (r=convert(dtype, 50),) # 50 nm radius

function unit_points(::Ring, shape; spacing::Real)
    circumference = 2π * shape.r
    npoints = ceil(Int, circumference / spacing)
    return unit_ring(typeof(shape.r), npoints)
end

local_points(::Ring, unit_ring, shape) = shape.r .* unit_ring

# the radius, with the default prior on sizes
function shape_quantities(::Ring, suffix)
    return (Symbol(:r, suffix) => (Size{Symbol(:r, suffix)}(), SIZE_PRIOR),)
end

shape_summary(::Ring, shape) = (r=shape.r,)
