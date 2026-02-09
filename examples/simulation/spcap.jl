using GLMakie
using CSV
using DataFrames
using Random
using LocMoFit
using Distributions

rng = Random.default_rng()
Random.seed!(rng, 1)

model = SphericalCap()
dtype = Float64

ps = (
    x=convert(dtype, 0),
    y=convert(dtype, 0),
    z=convert(dtype, 0),
    c=convert(dtype, 0.02), # 50 nm radius
    α=convert(dtype, π / 2),
    θ=convert(dtype, 0),
    ϕ=convert(dtype, 0)
)

umodel = LocMoFit.unit_model(model, ps; spacing=3.0)
pts, _ = model(umodel, ps, ())

locs, loc_precs = simulate_localizations(rng, pts, psf_σ=25.0, mean_photons_per_event=20.0, mean_events_per_dye=3.0, labeling_efficiency=0.1, z_factor=1.0)

fig = Figure()
ax = Axis3(fig[1, 1], aspect=:data)
@views scatter!(ax, pts[1, :], pts[2, :], pts[3, :])
@views scatter!(ax, locs[1, :], locs[2, :], locs[3, :])
fig

CSV.write("simulated_data_spcap.csv", DataFrame(x=locs[1, :], y=locs[2, :], z=locs[3, :], σx=loc_precs[1, :], σy=loc_precs[2, :], σz=loc_precs[3, :]))
CSV.write("ground_truth_spcap.csv", DataFrame(x=pts[1, :], y=pts[2, :], z=pts[3, :]))