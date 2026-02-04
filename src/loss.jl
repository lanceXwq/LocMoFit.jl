"""
    pairwise_sqdist(X, Y)

Compute squared Euclidean distances between all pairs of points in `X` and `Y`.

# Examples
```jldoctest
julia> X = [1.0 3.0 5.0; 2.0 4.0 6.0]; Y = [1.0 3.0; 2.0 4.0];

julia> LocMoFit.pairwise_sqdist(X,Y)
2×3 Matrix{Float64}:
 0.0  8.0  32.0
 8.0  0.0   8.0
```
"""
function pairwise_sqdist(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    X² = sum(abs2, X; dims=1)
    Y² = sum(abs2, Y; dims=1)
    Y²' .+ X² .- 2 .* (Y' * X)
end

function pairwise_sqdist(X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    invΣ::AbstractMatrix{T}) where {T}
    D² = muladd((Y .^ 2)', invΣ, sum(invΣ .* X .^ 2; dims=1))
    mul!(D², Y', invΣ .* X, -T(2), one(T))
end

negative_log_likelihood(datapoints, modelpoints, half_τ, log_const) = negative_log_likelihood(datapoints, hcat(modelpoints...), half_τ, log_const)

function negative_log_likelihood(datapoints::AbstractMatrix{T}, modelpoints::AbstractMatrix{T}, half_τ::T, log_const::T) where {T}
    Δx² = pairwise_sqdist(datapoints, modelpoints)
    per_pair_lls = muladd.(Δx², -half_τ, log_const)
    logsumexp(per_pair_lls; dims=1) |> sum |> (x -> -x) # may need a vec() in the future
end

function negative_log_likelihood(datapoints::AbstractMatrix{T}, modelpoints::AbstractMatrix{T}, half_τ::AbstractMatrix{T}, log_const::AbstractMatrix{T}) where {T}
    Δx² = pairwise_sqdist(datapoints, modelpoints, half_τ)
    per_pair_lls = log_const .- Δx²
    logsumexp(per_pair_lls; dims=1) |> sum |> (x -> -x) # may need a vec() in the future
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

function half_precision(stddev::AbstractArray)
    halfτ = similar(stddev)
    @. halfτ = stddev^-2 / 2
    return halfτ
end

function log_norm_const(stddev::AbstractArray)
    C = similar(stddev, 1, axes(stddev, 2))
    C .= -0.5 .* log.(prod(abs2, stddev, dims=1))
    return C
end
