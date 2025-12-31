module LocMoFit

using LuxCore
using NNlib: logsumexp
using Random
using Distances: SqMahalanobis, pairwise
using Rotations: RotZ, AngleAxis
using LinearAlgebra

include("models.jl")
include("loss.jl")

const setup = LuxCore.setup

export NPCLayer, NPC3DLayer, luxloss, optloss

end
