using CSV
using DataFrames
using LocMoFit

data = CSV.read(joinpath(@__DIR__, "..", "data", "ring.csv"), DataFrame)
locs = permutedims(Matrix(data[:, [:x_nm, :y_nm, :z_nm]]))
loc_precs = permutedims(Matrix(data[:, [:x_err_nm, :y_err_nm, :z_err_nm]]))
# the tutorial data's background fills a 300 nm cube around the site (examples/data/cases.jl)
fit = run_locmofit(
    "Ring",
    locs,
    loc_precs;
    max_iter=200,
    dtype=Float32,
    spacing=3.0,
    init_params=(x=15, y=0, z=25, r=57.5, theta=0.45, phi=0.95),
    freeze=(),
    fov_volume=300.0^3,
    history=true,
)
println(fit.loss)

ground_truth = filter(
    :is_signal => identity,
    CSV.read(joinpath(@__DIR__, "..", "data", "ring_truth.csv"), DataFrame),
)
include("../plotting/plotting functions.jl")
generate_video(;
    data=locs,
    model=fit.model,
    unitmodel=fit.unit_model,
    ground_truth=ground_truth,
    states=fit.history.parameters,
    losses=fit.history.losses,
    filename="fitting_process_ring.mp4",
    framerate=floor(Int, length(fit.history.losses) / 5),
)
