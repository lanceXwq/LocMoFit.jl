module LocMoFit

using Lux
using NNlib: logsumexp
using Random
using Rotations: RotZ, AngleAxis, RotationVec
using LinearAlgebra
using ComponentArrays: ComponentArray, ComponentVector
using Statistics: mean

using Enzyme
using Optimization
using OptimizationOptimJL

include("models.jl")
include("spcap.jl")
include("ring.jl")
include("double_ring.jl")
include("spcap_ring.jl")
include("unit models.jl")
include("transformations.jl")
include("loss.jl")

export Ring, SphericalCap, DoubleRing, SpCapRing, luxloss, optloss, run_locmofit, simulate_localizations

function run_locmofit(
    model_name::String,
    locs::AbstractMatrix{<:Real},
    loc_precisions::AbstractMatrix{<:Real};
    init_params::NamedTuple=NamedTuple(),
    freeze=(),
    max_iter::Integer=200,
    spacing::Real=3.0,
    dtype::Type{<:AbstractFloat}=Float32,
    seed=1,
)
    # @assert size(locs, 1) == 3 "locs must be (3,N), got $(size(locs))"
    @assert size(loc_precisions) == size(locs) "loc_precisions must match locs shape; got $(size(loc_precisions)) vs $(size(locs))"

    freeze_syms =
        freeze isa Symbol ? (freeze,) :
        freeze isa AbstractString ? (Symbol(freeze),) :
        freeze isa Tuple ? freeze :
        Tuple(freeze)

    freeze_syms = map(Symbol, freeze_syms)

    rng = Random.Xoshiro(seed)

    model = build_model(model_name)
    ps = apply_init(initialparameters(model, dtype), init_params)
    st = Lux.initialstates(rng, model)

    umodel = unit_model(model, ps, spacing=spacing)
    umodel_copy = umodel isa AbstractMatrix ? copy(umodel) : map(m -> copy(m), umodel)

    fmodel, ps, st = if isempty(freeze_syms)
        (model, ps, st)
    else
        Lux.Experimental.freeze(model, ps, st, freeze_syms)
    end

    locsT = locs isa Array{dtype} ? locs : Array{dtype}(locs)
    precT = loc_precisions isa Array{dtype} ? loc_precisions : Array{dtype}(loc_precisions)

    p = (
        model=fmodel,
        st=st,
        data=(locsT, precT),
        unit_model=umodel
    )

    losses = Vector{dtype}()
    states = []

    function callback(state, l)
        # state.objective is the current loss value
        if umodel != umodel_copy
            error("unit_model mutated at iter=$(state.iter)")
        end
        push!(losses, dtype(l))
        push!(states, NamedTuple(fullparams(state.u, st)))
        return false   # return true to stop early if you want
    end

    opt_fun = OptimizationFunction(optloss, AutoEnzyme())
    prob = OptimizationProblem(opt_fun, ComponentArray(ps), p)
    res = solve(prob, LBFGS(); callback, maxiters=max_iter)

    return (
        result=res,
        losses=losses,
        states=states,
        model=model,
        unit_model=umodel,
    )
end

function simulate_localizations(rng, pts; psf_σ, mean_photons_per_event, mean_events_per_dye, labeling_efficiency, z_factor=2.5)
    error("Simulation functions are defined in LocMoFitDistributionsExt, please load Distributions first")
end

end