function pairwise_distances²(𝐱_data::AbstractMatrix{T}, 𝐱_model::AbstractMatrix{T}) where {T}
    data_norms = sum(abs2, 𝐱_data; dims=2)
    model_norms = sum(abs2, 𝐱_model; dims=2)
    model_norms .+ data_norms' .- 2 .* (𝐱_model * 𝐱_data')
end

function negative_log_likelihood(datapoints::AbstractMatrix{T}, modelpoints::AbstractMatrix{T}, half_τ::T, log_const::T) where {T}
    distances² = pairwise_distances²(datapoints, modelpoints)
    per_pair_lls = muladd.(-distances², half_τ, log_const)
    -sum(logsumexp(per_pair_lls; dims=1)) # may need a vec() in the future
end

function luxloss(model, ps, st, data)
    positions, st = model((), ps, st)
    loss = negative_log_likelihood(data, positions, model.half_precision, model.log_const)
    return loss, st, NamedTuple()
end

function optloss(θ, p)
    # `p` is whatever you pass into OptimizationProblem
    model = p.model
    st0 = p.st
    datapoints = p.data
    modelpoints, _ = model((), θ, st0)
    return negative_log_likelihood(datapoints, modelpoints,
        model.half_precision, model.log_const)
end

