# const SPCAP_PARAMS = (:x, :y, :z, :c, :α, :θ, :ϕ)
# paramnames(::SphericalCap) = SPCAP_PARAMS
abstract type ModelParams{T} end

abstract type Model <: Lux.AbstractLuxLayer end

build_model(model_name::String) = getfield(LocMoFit, Symbol(model_name))()

@inline function fullparams(ps, st)
    return hasproperty(st, :frozen_params) ? Lux.Utils.merge(ps, st.frozen_params) : ps
end

function apply_init!(p::ComponentVector, init::NamedTuple)
    allowed = keys(p)  # tuple of Symbols
    for k in keys(init)
        if !(k in allowed)
            @warn "Ignoring unknown init key" key = k allowed = allowed
            continue
        end
        setproperty!(p, k, getproperty(init, k))
    end
    return p
end