module LocMoFit

using LuxCore
using NNlib: logsumexp
using Random
using Distances: SqMahalanobis, pairwise
using Rotations: RotZ, AngleAxis
using LinearAlgebra
using ComponentArrays: ComponentArray

include("basic shapes.jl")
include("composite shapes.jl")
include("transformations.jl")

include("models.jl")
include("loss.jl")

const setup = LuxCore.setup

export Ring, NuclearPoreComplex, luxloss, optloss

end
