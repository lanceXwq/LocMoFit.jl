using LocMoFit
using Test
using Random

rng = Random.default_rng()
Random.seed!(rng, 1)

@testset "LocMoFit.jl" begin
    x = randn(rng, Float32, 100, 3)
    y = randn(rng, Float32, 8, 3)
    Δx² = LocMoFit.pairwise_distances²(x, y)
    @test size(Δx²) == (8, 100)
    @test Δx² ≈ sum((insertdims(x, dims=1) .- insertdims(y, dims=2)) .^ 2; dims=3)
end
