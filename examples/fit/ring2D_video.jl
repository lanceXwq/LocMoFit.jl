using CSV
using DataFrames
using LocMoFit

data = CSV.read(joinpath(@__DIR__, "..", "data", "ring2D.csv"), DataFrame)
locs = permutedims(Matrix(data[:, [:x_nm, :y_nm]]))
loc_precs = permutedims(Matrix(data[:, [:x_err_nm, :y_err_nm]]))
# the tutorial data's background fills a 300 nm cube around the site (examples/data/cases.jl)
fit = run_locmofit(
    "Ring",
    locs,
    loc_precs;
    init_params=(x=15, y=0, z=20, r=57.5, theta=0.45, phi=0.95),
    freeze=(:z),
    fov_volume=300.0^2,
    history=true,
)
println(fit.loss)

# the same localizations in 3D (ring2D is the ring site without its z columns), for the video
data_3D = CSV.read(joinpath(@__DIR__, "..", "data", "ring.csv"), DataFrame)
locs_3D = permutedims(Matrix(data_3D[:, [:x_nm, :y_nm, :z_nm]]))
ground_truth = filter(
    :is_signal => identity,
    CSV.read(joinpath(@__DIR__, "..", "data", "ring_truth.csv"), DataFrame),
)
include("../plotting/plotting functions.jl")
generate_video(;
    data=locs_3D,
    model=fit.model,
    unitmodel=fit.unit_model,
    ground_truth=ground_truth,
    states=fit.history.parameters,
    losses=fit.history.losses,
    filename="fitting_process_ring2D.mp4",
    framerate=floor(Int, length(fit.history.losses) / 5),
)
