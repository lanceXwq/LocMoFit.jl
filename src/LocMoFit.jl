module LocMoFit

using Lux
using Random
using Rotations: RotZ, AngleAxis
using LinearAlgebra

include("models.jl")
include("loss.jl")

export NPCLayer, luxloss, optloss

end
