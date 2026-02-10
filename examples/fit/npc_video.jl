using CSV
using DataFrames
using LocMoFit

data = Matrix(CSV.read("simulated_data_npc.csv", DataFrame))
locs = permutedims(data[:, 1:3])
loc_precs = permutedims(data[:, 4:end])
res, losses, states, model, umodel = run_locmofit("NuclearPoreComplex", locs, loc_precs; max_iter=200, dtype=Float32, spacing=3.0, init_params=(z2=10, r1=10, r2=1, θ2=0, ϕ2=0), freeze=(:r2, :θ2, :ϕ2))
println(losses[end])

ground_truth = CSV.read("ground_truth_npc.csv", DataFrame)
include("../plotting/plotting functions.jl")
generate_video(; data=locs, model=model, unitmodel=umodel, ground_truth=ground_truth, states=states, losses=losses, filename="fitting_process_npc.mp4", framerate=floor(Int, length(losses) / 5))
