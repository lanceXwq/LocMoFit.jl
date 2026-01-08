module LocMoFit

using Lux
using NNlib: logsumexp
using Random
using Distances: SqMahalanobis, pairwise
using Rotations: RotZ, AngleAxis, Rotation
using LinearAlgebra
using ComponentArrays: ComponentArray

include("basic shapes.jl")
include("composite shapes.jl")
include("transformations.jl")

include("models.jl")
include("loss.jl")

const setup = Lux.setup
const share_parameters = Lux.Experimental.share_parameters
const Training = Lux.Training

export NPCLayer, NPC3DLayer, luxloss, optloss

end
