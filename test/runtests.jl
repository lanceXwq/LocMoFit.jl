using LocMoFit
using Test
using Random

rng = Random.default_rng()
Random.seed!(rng, 1)

@testset "LocMoFit.jl" begin
    x = randn(rng, Float32, 3, 100)
    y = randn(rng, Float32, 3, 8)
    Δx² = LocMoFit.pairwise_distances²(x, y)
    @test size(Δx²) == (8, 100)
    if VERSION ≥ v"1.12"
        @test Δx² ≈ sum((insertdims(x, dims=2) .- insertdims(y, dims=3)) .^ 2; dims=1)
    else
        X = copyto!(similar(x, 3, 1, 100), x)
        Y = copyto!(similar(y, 3, 8, 1), y)
        @test Δx² ≈ dropdims(sum((X .- Y) .^ 2; dims=1), dims=1)
    end
end
