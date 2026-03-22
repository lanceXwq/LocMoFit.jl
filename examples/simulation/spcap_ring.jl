using GLMakie
using CSV
using DataFrames
using Random
using LocMoFit
using Distributions

rng = Random.default_rng()
Random.seed!(rng, 1)

model = SpCapRing()
dtype = Float64

ps = (
    x1=convert(dtype, 0),
    y1=convert(dtype, 0),
    z1=convert(dtype, 0),
    r1=convert(dtype, 50), # 50 nm radius
    θ1=convert(dtype, π / 8),
    ϕ1=convert(dtype, π / √2),
    x2=convert(dtype, 0),
    y2=convert(dtype, 0),
    z2=convert(dtype, 50),
    c2=convert(dtype, 1), # 50 nm radius
    α2=convert(dtype, π / 2),
    θ2=convert(dtype, 0),
    ϕ2=convert(dtype, 0)
)

umodel = LocMoFit.unit_model(model, ps, spacing=10)
pts, _ = model(umodel, ps, ())

locs, loc_precs = simulate_localizations(rng, pts, psf_σ=0.21 * 642 / 1.35, mean_photons_per_event=100, mean_events_per_dye=5, labeling_efficiency=1.0, z_factor=1.0)

fig = Figure()
ax = Axis3(fig[1, 1], aspect=:data)
@views scatter!(ax, pts[1, :], pts[2, :], pts[3, :])
@views scatter!(ax, locs[1, :], locs[2, :], locs[3, :])
fig

CSV.write("simulated_data_scr.csv", DataFrame(x=locs[1, :], y=locs[2, :], z=locs[3, :], σx=loc_precs[1, :], σy=loc_precs[2, :], σz=loc_precs[3, :]))
CSV.write("ground_truth_scr.csv", DataFrame(x=pts[1, :], y=pts[2, :], z=pts[3, :]))