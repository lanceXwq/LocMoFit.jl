using CSV
using DataFrames
using LocMoFit

data = CSV.read(joinpath(@__DIR__, "..", "data", "double_ring.csv"), DataFrame)
locs = permutedims(Matrix(data[:, [:x_nm, :y_nm, :z_nm]]))
loc_precs = permutedims(Matrix(data[:, [:x_err_nm, :y_err_nm, :z_err_nm]]))
# the tutorial data's background fills a 300 nm cube around the site (examples/data/cases.jl)
fit = run_locmofit(
    "DoubleRing",
    locs,
    loc_precs;
    params=(
        x1=ParamToFit(5),
        y1=ParamToFit(5),
        z1=ParamToFit(-20),
        r1=ParamToFit(57.5),
        axis1=ParamToFit((0.35, 0.65)),
        z2=ParamToFit(55),
        r2=ParamToFit(Tied(:r1)),
        axis2=ParamToFit((0, 0), Fixed()),
    ),
    fov_volume=300.0^3,
    history=true,
)
println(fit.loss)

ground_truth = filter(
    :is_signal => identity,
    CSV.read(joinpath(@__DIR__, "..", "data", "double_ring_truth.csv"), DataFrame),
)
include("../plotting/plotting functions.jl")
generate_video(;
    data=locs,
    model=fit.model,
    unitmodel=fit.unit_model,
    ground_truth=ground_truth,
    states=fit.history.parameters,
    losses=fit.history.losses,
    filename="fitting_process_npc.mp4",
    framerate=floor(Int, length(fit.history.losses) / 5),
)
