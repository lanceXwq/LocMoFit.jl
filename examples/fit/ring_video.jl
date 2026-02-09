using CSV
using DataFrames
using LocMoFit

data = Matrix(CSV.read("simulated_data_ring.csv", DataFrame))
locs = permutedims(data[:, 1:3])
loc_precs = permutedims(data[:, 4:end])
res, losses, states, model, umodel = run_locmofit("Ring", locs, loc_precs; max_iter=200, dtype=Float32, spacing=3.0, init_params=(z=53, r=10), freeze=())
println(losses[end])

ground_truth = CSV.read("ground_truth_ring.csv", DataFrame)
include("../plotting/plotting functions.jl")
generate_video(; data=locs, model=model, unitmodel=umodel, ground_truth=ground_truth, states=states, losses=losses, filename="fitting_process_ring.mp4", framerate=floor(Int, length(losses) / 5))
