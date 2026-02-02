module LocMoFit

using Lux
using NNlib: logsumexp
using Random
using Distances: SqMahalanobis, pairwise
using Rotations: RotZ, AngleAxis, RotationVec
using LinearAlgebra
using ComponentArrays: ComponentArray, ComponentVector

using Enzyme
using Optimization
using OptimizationOptimJL

include("models.jl")
include("basic shapes.jl")
include("spcap.jl")
include("composite shapes.jl")
include("unit models.jl")
include("transformations.jl")
include("loss.jl")

export Ring, SphericalCap, NuclearPoreComplex, luxloss, optloss, run_locmofit

function run_locmofit(model_name::String, locs::AbstractMatrix{<:Real}, loc_precisions::AbstractMatrix{<:Real}; seed=1, max_iter::Integer=200, dtype::Type{<:AbstractFloat}=Float32, init::NamedTuple=NamedTuple(), freeze::Tuple{Vararg{Symbol}}=Tuple{}(), spacing::Real=3.0)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    model = build_model(model_name)
    ps = initialparameters(model, dtype)
    st = Lux.initialstates(rng, model)
    apply_init!(ps, init)

    umodel = unit_model(model, ps, spacing=spacing)
    umodel_copy = copy(umodel)

    if isempty(freeze)
        fmodel, ps, st = model, ps, st
    else
        fmodel, ps, st = Lux.Experimental.freeze(model, ps, st, freeze)
        ps = ComponentArray(ps)
    end

    locs = convert(Array{dtype}, locs)
    loc_precisions = convert(Array{dtype}, loc_precisions)

    p = (
        model=fmodel,
        st=st,
        data=locs,
        half_precision=half_precision(loc_precisions, dtype=dtype),
        log_const=log_norm_const(loc_precisions; dtype=dtype),
        unit_model=umodel
    )

    losses = dtype[]
    states = []

    function callback(state, l)
        # state.objective is the current loss value
        if umodel != umodel_copy
            error("unit_model mutated at iter=$(state.iter)")
        end
        push!(losses, state.objective)
        push!(states, fullparams(state.u, st))
        return false   # return true to stop early if you want
    end

    opt_fun = OptimizationFunction(optloss, AutoEnzyme())
    prob = OptimizationProblem(opt_fun, ps, p)
    res = solve(prob, LBFGS(); callback, maxiters=max_iter)

    return res, losses, states, model, umodel
end

end