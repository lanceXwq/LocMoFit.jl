# const SPCAP_PARAMS = (:x, :y, :z, :c, :α, :θ, :ϕ)
# paramnames(::SphericalCap) = SPCAP_PARAMS
abstract type ModelParams{T} end

abstract type Model <: Lux.AbstractLuxLayer end

build_model(model_name::String) = getfield(LocMoFit, Symbol(model_name))()

@inline function fullparams(ps, st)
    return hasproperty(st, :frozen_params) ? Lux.Utils.merge(ps, st.frozen_params) : ps
end

function apply_init(default::NamedTuple, init::NamedTuple)
    allowed = keys(default)
    init_keys = keys(init)
    isempty(init_keys) && return default

    init_set = Set(init_keys)

    unknown = Tuple(k for k in init_keys if !(k in allowed))
    if !isempty(unknown)
        @warn "Ignoring unknown init keys" unknown = unknown allowed = allowed
    end

    vals = ntuple(length(allowed)) do i
        k = allowed[i]
        old = getproperty(default, k)
        k in init_set || return old

        new = getproperty(init, k)

        if old isa Number
            new isa Number || throw(ArgumentError("init[$k] must be a Number; got $(typeof(new))"))
            return convert(typeof(old), new)
        elseif old isa AbstractArray
            new isa AbstractArray || throw(ArgumentError("init[$k] must be an AbstractArray; got $(typeof(new))"))
            size(new) == size(old) || throw(ArgumentError("init[$k] has wrong size: expected $(size(old)), got $(size(new))"))

            out = similar(old, eltype(old), axes(old))
            out .= new
            return out
        else
            throw(ArgumentError("default[$k] must be Number or AbstractArray; got $(typeof(old))"))
        end
    end

    return NamedTuple{allowed}(vals)
end