"""
    pairwise_distances²(X, Y)

Compute squared Euclidean distances between all pairs of points in `X` and `Y`.

# Examples
```jldoctest
julia> X = [1.0 3.0 5.0; 2.0 4.0 6.0]; Y = [1.0 3.0; 2.0 4.0];

julia> LocMoFit.pairwise_distances²(X,Y)
2×3 Matrix{Float64}:
 0.0  8.0  32.0
 8.0  0.0   8.0
```
"""
function pairwise_distances²(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    X² = sum(abs2, X; dims=1)
    Y² = sum(abs2, Y; dims=1)
    Y²' .+ X² .- 2 .* (Y' * X)
end

negative_log_likelihood(datapoints::AbstractMatrix{T}, modelpoints, half_τ::T, log_const::T) where {T} = negative_log_likelihood(datapoints, hcat(modelpoints...), half_τ, log_const)

function negative_log_likelihood(datapoints::AbstractMatrix{T}, modelpoints::AbstractMatrix{T}, half_τ::T, log_const::T) where {T}
    Δx² = pairwise_distances²(datapoints, modelpoints)
    per_pair_lls = muladd.(-Δx², half_τ, log_const)
    -sum(logsumexp(per_pair_lls; dims=1)) # may need a vec() in the future
end

function luxloss(model, ps, st, data)
    positions, st = model(data.in, ps, st)
    loss = negative_log_likelihood(data.out, positions, data.half_precision, data.log_const)
    return loss, st, NamedTuple()
end

function optloss(θ, p)
    # `p` is whatever you pass into OptimizationProblem
    model = p.model
    st = p.st
    datapoints = p.data
    modelpoints, _ = model(p.unit_model, θ, st)
    return negative_log_likelihood(datapoints, modelpoints,
        p.half_precision, p.log_const)
end

