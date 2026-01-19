module LocMoFit

using LuxCore
using LuxLib
using NNlib: logsumexp
using Random
using Distances: SqMahalanobis, pairwise
using Rotations: RotZ, AngleAxis, RotationVec
using LinearAlgebra
using ComponentArrays: ComponentArray

include("basic shapes.jl")
include("composite shapes.jl")
include("unit models.jl")
include("transformations.jl")
include("loss.jl")

# const setup = LuxCore.setup

export Ring, SphericalCap, NuclearPoreComplex, luxloss, optloss

end
