using GLMakie
using CSV
using DataFrames
using Random
using LocMoFit
using Distributions

rng = Random.default_rng()
Random.seed!(rng, 1)

model = Ring()
dtype = Float64

ps = (
    x=convert(dtype, 0),
    y=convert(dtype, 0),
    z=convert(dtype, 0),
    r=convert(dtype, 50), # 50 nm radius
    θ=convert(dtype, π / 8),
    ϕ=convert(dtype, π / √2)
)

umodel = LocMoFit.unit_model(model, ps, 8)
pts, _ = model(umodel, ps, ())

locs, loc_precs = simulate_localizations(rng, pts, psf_σ=0.21 * 642 / 1.35, mean_photons_per_event=100, mean_events_per_dye=5, labeling_efficiency=1.0, z_factor=1.0)

fig = Figure()
ax = Axis3(fig[1, 1], aspect=:data)
@views scatter!(ax, pts[1, :], pts[2, :], pts[3, :])
@views scatter!(ax, locs[1, :], locs[2, :], locs[3, :])
fig

CSV.write("simulated_data_ring.csv", DataFrame(x=locs[1, :], y=locs[2, :], z=locs[3, :], σx=loc_precs[1, :], σy=loc_precs[2, :], σz=loc_precs[3, :]))
CSV.write("ground_truth_ring.csv", DataFrame(x=pts[1, :], y=pts[2, :], z=pts[3, :]))