using LocMoFit
using Test
using Random
using Distances: pairwise!, SqEuclidean, SqMahalanobis
using LinearAlgebra: Diagonal

rng = Random.default_rng()
Random.seed!(rng, 1)

@testset "pairwise squared distances" begin
    N, M = 10, 20
    x = randn(rng, Float32, 3, N)
    y = randn(rng, Float32, 3, M)
    sqeuclidean = LocMoFit.pairwise_sqdist(x, y)
    @test size(sqeuclidean) == (M, N)
    if VERSION ≥ v"1.12"
        @test sqeuclidean ≈ sum((insertdims(x, dims=2) .- insertdims(y, dims=3)) .^ 2; dims=1)
    else
        X = copyto!(similar(x, 3, 1, N), x)
        Y = copyto!(similar(y, 3, M, 1), y)
        @test sqeuclidean ≈ dropdims(sum((X .- Y) .^ 2; dims=1), dims=1)
    end

    σ⁻² = rand(rng, Float32, 3, N)
    sqmahalabonis1 = similar(sqeuclidean)
    for i in axes(x, 2)
        @views pairwise!(SqMahalanobis(Diagonal(σ⁻²[:, i])), sqmahalabonis1[:, i:i], y, x[:, i:i], dims=2)
    end
    @test sqmahalabonis1 ≈ LocMoFit.pairwise_sqdist(x, y, σ⁻²)
end
