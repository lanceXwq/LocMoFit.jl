# How often a fit misses the optimum from the starting values estimated from the data (estimate_start) and from the model's defaults, on the sites of generate.jl, each with its own truth (derivations/derivations.qmd, Starting values from the data; DESIGN.md). A fit misses when its final loss exceeds that of the fit started from the truth by more than 0.01 per localization. Also printed: the estimated axis's angle from the true one, and the time of the estimate and of the fits. Run from the LocMoFit.jl folder, after generate.jl:
#     julia --project=. benchmarks/start/fit.jl quick     # the first 20 sites of every group, after a change to the estimate: about 2 minutes
#     julia --project=. benchmarks/start/fit.jl           # every site: about 5 minutes
#     julia --project=. benchmarks/start/fit.jl SpCapRing # the groups whose label starts with this
using LocMoFit, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "..", "test", "tutorial.jl"))
const DATA = joinpath(@__DIR__, "data")
const QUICK = "quick" in ARGS
const ONLY = filter(!=("quick"), ARGS)
const FOV = 300.0^3   # the field of view of the simulation (examples/data/cases.jl)

# every site of the case `name`, as (localizations, precisions, true parameters)
function sites(name)
    L = readcsv(joinpath(DATA, "$(name)_locs.csv"))
    T = readcsv(joinpath(DATA, "$(name)_truth.csv"))
    names = filter(!=(:site), keys(T))
    n = QUICK ? min(20, length(T.site)) : length(T.site)
    return map(1:n) do j
        rows = findall(==(T.site[j]), L.site)
        locs = matrix(L, (:x_nm, :y_nm, :z_nm), rows)
        precs = matrix(L, (:x_err_nm, :y_err_nm, :z_err_nm), rows)
        return (locs, precs, NamedTuple{names}(Tuple(T[k][j] for k in names)))
    end
end

axis_of(p, k) = LocMoFit.axis_vector(p[Symbol(:theta, k)], p[Symbol(:phi, k)])

# the model's default starting values in the names start_params takes: every axis along z (theta = phi = 0 rather than the chart coordinates vx, vy), and the default fractions, a background fraction of 0.1 and equal shares (ω₀ = log(f₀/f₁) with f₁ = 0.9/K for K components)
function default_start(m)
    d = LocMoFit.default_parameters(m, Float64)
    shape = NamedTuple(k => v for (k, v) in pairs(d) if !startswith(string(k), "v"))
    axes = LocMoFit.orientation_names(m)
    K = length(axes)
    return merge(
        shape, NamedTuple(n => 0.0 for n in Iterators.flatten(axes)), (omega0=log(K / 9),)
    )
end

# `k`: the suffix of the first component's parameters ("" for a simple model, 1 for a composite); `fixed`: the quantities held at their starting values
function run(label, name, model; fixed=(), k="")
    isempty(ONLY) || any(o -> startswith(label, o), ONLY) || return nothing
    S = sites(name)
    m = LocMoFit.build_model(model)
    defaults = start_params(default_start(m); fixed)
    held = NamedTuple(q => ParamToFit(Fixed()) for q in fixed)
    missed_estimate, missed_defaults, axis_error = 0, 0, Float64[]
    t_estimate, t_fits_estimate, t_fits_defaults = 0.0, 0.0, 0.0
    run_locmofit(model, S[1][1], S[1][2]; params=held, fov_volume=FOV)   # compile
    for (locs, precs, truth) in S
        t_estimate += @elapsed est = LocMoFit.estimate_start(m, locs, precs, nothing)
        c = abs(dot(axis_of(est, k), axis_of(truth, k)))
        push!(axis_error, acosd(min(c, 1)))
        truth_params = start_params(truth; fixed)
        reference =
            run_locmofit(model, locs, precs; params=truth_params, fov_volume=FOV).loss
        t_fits_estimate += @elapsed a = run_locmofit(
            model, locs, precs; params=held, fov_volume=FOV
        )
        t_fits_defaults += @elapsed d = run_locmofit(
            model, locs, precs; params=defaults, fov_volume=FOV
        )
        missed_estimate += a.loss > reference + 0.01
        missed_defaults += d.loss > reference + 0.01
    end
    n = length(S)
    @printf(
        "%-44s %3d sites: missed %3d from the estimate, %3d from the defaults; estimate's axis off median %4.1f deg, > 45 deg on %2d; estimate %5.2f ms per site; fits %5.1f s (estimate) vs %5.1f s (defaults)\n",
        label,
        n,
        missed_estimate,
        missed_defaults,
        sort(axis_error)[(n + 1) ÷ 2],
        count(>(45), axis_error),
        1e3t_estimate / n,
        t_fits_estimate,
        t_fits_defaults
    )
end

const COAXIAL = (:x2, :y2, :axis2)
run("Ring", "ring", "Ring")
run("SphericalCap", "spcap", "SphericalCap")
run(
    "DoubleRing, coaxial (x2, y2, axis2 fixed)",
    "double_ring",
    "DoubleRing";
    fixed=COAXIAL,
    k=1,
)
run("DoubleRing, all free", "double_ring", "DoubleRing"; k=1)
run(
    "SpCapRing, coaxial (x2, y2, axis2 fixed)",
    "spcap_ring",
    "SpCapRing";
    fixed=COAXIAL,
    k=1,
)
run("SpCapRing, all free", "spcap_ring", "SpCapRing"; k=1)
