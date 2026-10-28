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

shape_penalty(::Ring, shape) = sq_hinge(1 / shape.r^2 - 1)

shape_summary(::Ring, shape) = (r=shape.r,)
