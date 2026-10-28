# Accuracy of LocMoFit.jl's fits: every simulated site of generate.jl is fitted, and the errors against the truth are summarized per model as bias (mean error), scatter (standard deviation over sites) and their ratio z = bias / SE, where SE = scatter / √(sites). A |z| above 3 flags a bias the scatter cannot explain. Run from the LocMoFit.jl folder, after generate.jl:
#     julia -t 8 --project=. benchmarks/accuracy/fit.jl quick     # the quick subset, after every change: about 1 minute
#     julia -t 1 --project=. benchmarks/accuracy/fit.jl            # every group, once in a while: about 2 minutes
#     julia -t 8 --project=. benchmarks/accuracy/fit.jl spcap 4    # one group (a case name and a setting number)
# The quick subset is every model in the realistic setting (background, moved start, Float64); most of its time is compilation. With 20 sites a flagged bias can still be a fluctuation: the cap's 20 sites give a low κ′ (z ≈ −3), while 60 fresh sites gave none (z ≈ 0.3, 2026-10-26). Confirm a flag with more sites. The sites are fitted in threads (`julia -t N`), which can crash Julia: it happened twice on 2026-10-26 ("GC error (probable corruption)") and once on 2026-10-27, after 22 of the 25 groups of a full run (memory corruption detected by malloc, "Illegal instruction"), while five cold-start runs and the quick subset with bounds checks on did not. The cause is not known. When a run crashes, rerun the missing groups one at a time with `julia -t 1`.
using LocMoFit, LinearAlgebra, Statistics, Printf
using LocMoFit: fit_problem, full_parameters
const DATA = joinpath(@__DIR__, "data")
const EXAMPLES = joinpath(@__DIR__, "..", "..", "examples", "data")

parsefield(s) =
    if s == "true"
        1.0
    elseif s == "false"
        0.0
    else
        parse(Float64, s)
    end
function readcsv(path)
    L = readlines(path)
    names = Tuple(Symbol.(split(L[1], ",")))
    rows = [parsefield.(split(l, ",")) for l in L[2:end]]
    return NamedTuple{names}(Tuple([r[k] for r in rows] for k in eachindex(names)))
end

angle_deg(a, b) = acosd(clamp(dot(a, b) / (norm(a) * norm(b)), -1, 1))
axis_deg(a, b) = min(angle_deg(a, b), 180 - angle_deg(a, b))      # a ring's axis has no sign
mirror(a) = [a[1], a[2], -a[3]]

# the true structure in the world frame, as run_locmofit reports a fitted one: started at the truth (the starting axes at the true ones, the chart coordinates at 0), the starting point is the truth
function true_components(model, truth, locs, precs)
    p, u = fit_problem(
        LocMoFit.build_model(model), locs, precs; init_params=truth, dtype=Float64
    )
    return LocMoFit.components(p.model, p.unit_model, full_parameters(u, p.parameter_map))
end

# what is compared, in the world frame: centers (nm), sizes, and axes (degrees); `model` is the model's name
function geometry(model, c)
    model == "Ring" && return (center=c[1].position, r=c[1].r, axis=c[1].axis)
    model == "SphericalCap" && return (
        apex=c[1].position,
        s=c[1].s,
        kappa=c[1].kappa,
        c=c[1].curvature,
        alpha=c[1].alpha,
        axis=c[1].axis,
    )
    model == "DoubleRing" && return (
        center1=c[1].position,
        r1=c[1].r,
        axis1=c[1].axis,
        center2=c[2].position,
        r2=c[2].r,
        axis2=c[2].axis,
    )
    # SpCapRing: the cap's sphere's center is compared, as in the baseline
    return (
        center=c[1].position,
        r=c[1].r,
        axis=c[1].axis,
        cap_center=c[2].position - c[2].radius * c[2].axis,
        cap_radius=c[2].radius,
        cap_alpha=c[2].alpha,
        cap_kappa=c[2].kappa,
        cap_axis=c[2].axis,
    )
end
# errors as named scalars: each vector becomes its components (nm), each axis its angle to the true one (°). In 2D a ring's axis and its mirror image through the xy plane look the same, so the smaller of the two angles counts.
function errors(model, fitted, truth; D=3)
    f, t = geometry(model, fitted), geometry(model, truth)
    out = Pair{String,Float64}[]
    for k in keys(t)
        name = string(k)
        if occursin("axis", name)
            compare = model == "SphericalCap" || name == "cap_axis" ? angle_deg : axis_deg
            push!(
                out,
                name * " (°)" => if D == 2
                    min(compare(f[k], t[k]), compare(f[k], mirror(t[k])))
                else
                    compare(f[k], t[k])
                end,
            )
        elseif t[k] isa AbstractVector
            append!(
                out,
                [
                    name * "." * c => f[k][i] - t[k][i] for
                    (i, c) in enumerate(("x", "y", "z"))
                ],
            )
        else
            push!(out, name => f[k] - t[k])
        end
    end
    return out
end

# the moved start: positions +5 nm, sizes ×1.15 (a cap's κ′ ×0.85), starting axes +0.15 rad in θ and ϕ, blur 2 nm, and the default signal fraction (0.9) and shares (equal)
function moved(p)
    shift(k, v) =
        if k in (:x, :y, :z, :x1, :y1, :z1, :x2, :y2, :z2)
            v + 5
        elseif k in (:r, :r1, :r2, :s, :s2)
            1.15v
        elseif k in (:eta, :eta2)
            atanh(0.85tanh(v))
        elseif k in (:theta, :phi, :theta1, :phi1, :theta2, :phi2)
            v + 0.15
        elseif k == :sigma
            2.0
        elseif k == :logit_prob
            log(9)
        elseif k == :omega2
            0.0
        else
            v
        end
    return NamedTuple{keys(p)}(Tuple(shift(k, v) for (k, v) in pairs(p)))
end

const CASES = (
    (name="ring", model="Ring", D=3, freeze=()),
    (name="ring2D", model="Ring", D=2, freeze=(:z,)),        # the ring sites without z; z is then undetermined
    (name="spcap", model="SphericalCap", D=3, freeze=()),
    (name="double_ring", model="DoubleRing", D=3, freeze=()),
    (name="spcap_ring", model="SpCapRing", D=3, freeze=()),
)
const SETTINGS = (
    (background=false, start=:truth, dtype=Float64),
    (background=false, start=:moved, dtype=Float64),
    (background=true, start=:truth, dtype=Float64),
    (background=true, start=:moved, dtype=Float64),
    (background=true, start=:moved, dtype=Float32),
)

function fit_sites(case, setting)
    file =
        (case.name == "ring2D" ? "ring" : case.name) *
        (setting.background ? "_background" : "_nobackground")
    d = readcsv(joinpath(DATA, file * ".csv"))
    labels = readcsv(joinpath(DATA, file * "_truth.csv"))
    truth = map(first, readcsv(joinpath(EXAMPLES, case.name * "_parameters.csv")))
    setting.background || (truth = merge(truth, (logit_prob=log(99),)))   # no background: start at 99% signal
    init = if setting.start === :truth
        truth
    else
        merge(moved(truth), NamedTuple{case.freeze}(map(k -> truth[k], case.freeze)))
    end
    cols = if case.D == 3
        ((:x_nm, :y_nm, :z_nm), (:x_err_nm, :y_err_nm, :z_err_nm))
    else
        ((:x_nm, :y_nm), (:x_err_nm, :y_err_nm))
    end
    sites = sort(unique(d.site))
    function fit_site(k)
        rows = findall(==(sites[k]), d.site)
        locs = permutedims(reduce(hcat, [d[c][rows] for c in cols[1]]))
        precs = permutedims(reduce(hcat, [d[c][rows] for c in cols[2]]))
        time = @elapsed r = run_locmofit(
            case.model,
            locs,
            precs;
            init_params=init,
            freeze=case.freeze,
            dtype=setting.dtype,
            fov_volume=300.0^case.D,
        )
        signal = labels.is_signal[rows] .> 0.5
        e = errors(
            case.model,
            r.components,
            true_components(case.model, truth, locs, precs);
            D=case.D,
        )
        push!(e, "sigma" => r.parameters.sigma)
        push!(e, "signal fraction" => r.signal_fraction - mean(signal))
        r.parameters isa NamedTuple &&
            haskey(r.parameters, :omega2) &&
            push!(e, "share 1" => r.weights[1] - mean(labels.component[rows][signal] .== 1))
        return (
            errors=e,
            retcode=string(r.result.retcode),
            iterations=r.result.stats.iterations,
            time,
        )
    end
    # the first site alone, so that the gradient is compiled before several threads need it (compiling it on several threads at once crashed Julia); its time includes the compilation
    results = Vector{Any}(undef, length(sites))
    results[1] = fit_site(1)
    Threads.@threads for k in 2:length(sites)
        results[k] = fit_site(k)
    end
    return results
end

function summarize(case, setting, results)
    @printf(
        "\n%s — %s, start at the %s, %s: %d sites, %d converged, median %.0f iterations, median %.0f ms per fit\n",
        case.name,
        setting.background ? "background" : "no background",
        setting.start,
        setting.dtype,
        length(results),
        count(r -> r.retcode == "Success", results),
        median(r.iterations for r in results),
        1e3median(r.time for r in results)
    )
    @printf("  %-16s %10s %10s %7s\n", "error", "bias", "scatter", "z")
    for (j, (name, _)) in enumerate(first(results).errors)
        e = [r.errors[j][2] for r in results]
        b, s = mean(e), std(e)
        flag = if occursin("(°)", name)
            ""
        elseif abs(b) > 3s / sqrt(length(e))
            "  ← bias"
        else
            ""
        end
        @printf(
            "  %-16s %10.4g %10.4g %7.1f%s\n", name, b, s, b / (s / sqrt(length(e))), flag
        )   # 4 digits: c is ~0.01
    end
end

case_named(name) = only(filter(c -> c.name == name, collect(CASES)))
groups = if isempty(ARGS)
    [(c, s) for c in CASES for s in SETTINGS]
elseif ARGS == ["quick"]
    [(c, SETTINGS[4]) for c in CASES]
else
    [(case_named(ARGS[1]), SETTINGS[parse(Int, ARGS[2])])]
end
for (case, setting) in groups
    results = fit_sites(case, setting)
    summarize(case, setting, results)
    flush(stdout)
end
