"""
    ParameterMap(θ₀, fixed, tie=())

The map from the fitted vector `u`, which holds only the free parameters, to the full parameter set θ = base + S·u. `base` holds the values of the fixed parameters (zero for the others), and `S` is a constant 0/1 matrix with one 1 per free or tied parameter. The map is a matrix-vector product, so it costs little and Enzyme differentiates it without trouble.

`θ₀` is the `NamedTuple` of starting values, `fixed` the names of the parameters to keep at them, and `tie` pairs `target => source`: the target takes the source's value throughout (its own starting value is ignored), through a 1 in the source's column of `S`. A target tied to a fixed parameter is fixed at that value; ties may chain.
"""
struct ParameterMap{names,free,T}
    base::Vector{T}
    S::Matrix{T}
end

function ParameterMap(
    θ₀::NamedTuple{names},
    fixed::Tuple{Vararg{Symbol}},
    tie::Tuple{Vararg{Pair{Symbol,Symbol}}}=(),
) where {names}
    targets = map(first, tie)
    unknown = Tuple(k for k in (fixed..., targets..., map(last, tie)...) if !(k in names))
    isempty(unknown) ||
        throw(ArgumentError("unknown parameters $unknown; the parameters are $names"))
    allunique(targets) || throw(ArgumentError("a parameter is tied twice: $tie"))
    both = Tuple(k for k in targets if k in fixed)
    isempty(both) || throw(ArgumentError("parameters both fixed and tied: $both"))
    # follow each tie back to a parameter that is not tied itself
    source = Dict(tie)
    function root(k, seen=Symbol[])
        haskey(source, k) || return k
        k in seen && throw(ArgumentError("the ties form a cycle: $tie"))
        return root(source[k], push!(seen, k))
    end
    free = Tuple(k for k in names if !(k in fixed) && !(k in targets))
    isempty(free) &&
        throw(ArgumentError("every parameter is fixed or tied; there is nothing to fit"))
    T = typeof(first(θ₀))
    base = T[root(k) in free ? zero(T) : θ₀[root(k)] for k in names]
    S = T[root(names[i]) == free[j] for i in eachindex(names), j in eachindex(free)]
    return ParameterMap{names,free,T}(base, S)
end

"""
    free_parameters(θ, map)

The free parameters of θ, as the `ComponentVector` the optimizer works on.
"""
function free_parameters(θ::NamedTuple, ::ParameterMap{names,free}) where {names,free}
    return ComponentVector(NamedTuple{free}(map(k -> θ[k], free)))
end

"""
    full_parameters(u, map)

All parameters, fixed and tied ones included, as a `NamedTuple` of scalars.
"""
function full_parameters(u::AbstractVector, m::ParameterMap{names}) where {names}
    θ = muladd(m.S, u, m.base)
    return NamedTuple{names}(ntuple(i -> θ[i], Val(length(names))))
end
