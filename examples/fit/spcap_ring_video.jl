using CSV
using DataFrames
using LocMoFit

data = CSV.read(joinpath(@__DIR__, "..", "data", "spcap_ring.csv"), DataFrame)
locs = permutedims(Matrix(data[:, [:x_nm, :y_nm, :z_nm]]))
loc_precs = permutedims(Matrix(data[:, [:x_err_nm, :y_err_nm, :z_err_nm]]))
# the tutorial data's background fills a 300 nm cube around the site (examples/data/cases.jl)
fit = run_locmofit(
    "SpCapRing",
    locs,
    loc_precs;
    max_iter=200,
    dtype=Float32,
    spacing=3.0,
    init_params=(
        x1=5, y1=5, z1=-25, r1=57.5, theta1=0.35, phi1=0.65, z2=95, s2=29, eta2=0.45
    ),
    freeze=(:vx2, :vy2),
    fov_volume=300.0^3,
    history=true,
)
println(fit.loss)

ground_truth = filter(
    :is_signal => identity,
    CSV.read(joinpath(@__DIR__, "..", "data", "spcap_ring_truth.csv"), DataFrame),
)
include("../plotting/plotting functions.jl")
generate_video(;
    data=locs,
    model=fit.model,
    unitmodel=fit.unit_points,
    ground_truth=ground_truth,
    states=fit.history.parameters,
    losses=fit.history.losses,
    filename="fitting_process_scr.mp4",
    framerate=floor(Int, length(fit.history.losses) / 5),
)
