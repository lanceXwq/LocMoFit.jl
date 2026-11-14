# Speed of LocMoFit.jl's fits. For every model on its tutorial site (examples/data), from the moved start of test/fit.jl: the loss, its Enzyme gradient and a complete fit, in Float64 and Float32. With `threads`, also the throughput of many sites fitted at once: the accuracy benchmark's 20 sites per model (benchmarks/accuracy/data, from its generate.jl), one after another and in threads. Run from the LocMoFit.jl folder, with one thread per performance core:
#     julia -t 8 --project=. benchmarks/speed/speed.jl [threads] [case names...]
# BenchmarkTools comes from the global environment. Enzyme's compiled code can vary between Julia sessions (DynMoFit.jl/NOTES.md), so compare timings across more than one run.
using LocMoFit, BenchmarkTools, Enzyme, Printf
using LocMoFit: loss
include(joinpath(@__DIR__, "..", "..", "test", "tutorial.jl"))

const ACCURACY = joinpath(@__DIR__, "..", "accuracy", "data")
const CASES = (
    (name="ring", model="Ring", D=3, freeze=()),
    (name="ring2D", model="Ring", D=2, freeze=(:z,)),
    (name="double_ring", model="DoubleRing", D=3, freeze=()),
    (name="spcap", model="SphericalCap", D=3, freeze=()),
    (name="spcap_ring", model="SpCapRing", D=3, freeze=()),
)

function start(case)
    truth = tutorial_parameters(case.name)
    return merge(moved(truth), NamedTuple{case.freeze}(map(k -> truth[k], case.freeze)))
end

function site_speed(case, dtype)
    d = readcsv(joinpath(TUTORIAL, case.name * ".csv"))
    cols = columns(case.D)
    locs, precs = matrix(d, cols[1]), matrix(d, cols[2])
    kw = (
        params=start_params(start(case); fixed=case.freeze), dtype, fov_volume=300.0^case.D
    )
    p, u = fit_problem(LocMoFit.build_model(case.model), locs, precs; kw...)
    N, M = size(locs, 2), sum(length, p.ranges)
    du = zero(u)
    Enzyme.autodiff(Reverse, loss, Active, Duplicated(u, du), Const(p))
    tv = @belapsed loss($u, $p) samples = 50 evals = 1 seconds = 2
    tg = @belapsed (
        fill!($du, 0);
        Enzyme.autodiff(Reverse, loss, Active, Duplicated($u, $du), Const($p))
    ) samples = 50 evals = 1 seconds = 2
    run_locmofit(case.model, locs, precs; kw...)
    tf = minimum(@elapsed(run_locmofit(case.model, locs, precs; kw...)) for _ in 1:3)
    r = run_locmofit(case.model, locs, precs; kw...)
    @printf(
        "%-12s %-8s %5d %6d %10.3g %10.3g %8.2f %8.2f %9.3g %5d %5d %5d  %s\n",
        case.name,
        dtype,
        N,
        M,
        1e3tv,
        1e3tg,
        1e9tv / (N * M),
        1e9tg / (N * M),
        tf,
        r.result.stats.iterations,
        r.result.stats.fevals,
        r.result.stats.gevals,
        r.result.retcode
    )
end

# fits of the accuracy benchmark's sites (background, moved start, Float64), one after another and in threads
function throughput(case)
    d = readcsv(
        joinpath(ACCURACY, (case.name == "ring2D" ? "ring" : case.name) * "_background.csv")
    )
    cols = columns(case.D)
    sites = sort(unique(d.site))
    data = map(
        s -> (
            rows=findall(==(s), d.site);
            (matrix(d, cols[1], rows), matrix(d, cols[2], rows))
        ),
        sites,
    )
    kw = (params=start_params(start(case); fixed=case.freeze), fov_volume=300.0^case.D)
    fit(k) = run_locmofit(case.model, data[k]...; kw...).loss
    fit(1)                                   # compile on one thread first
    serial = @elapsed foreach(fit, eachindex(data))
    losses = zeros(length(data))
    threaded = @elapsed Threads.@threads for k in eachindex(data)
        losses[k] = fit(k)
    end
    same = losses == map(fit, eachindex(data))
    @printf(
        "%-12s %5d sites %9.3g s %9.3g s %6.2f×   identical losses: %s\n",
        case.name,
        length(data),
        serial,
        threaded,
        serial / threaded,
        same
    )
end

function main(args)
    names = filter(!=("threads"), args)
    cases = isempty(names) ? CASES : filter(c -> c.name in names, CASES)
    println("LocMoFit at ", pkgdir(LocMoFit), "; ", Threads.nthreads(), " threads")
    @printf(
        "%-12s %-8s %5s %6s %10s %10s %8s %8s %9s %5s %5s %5s\n",
        "case",
        "dtype",
        "N",
        "M",
        "value ms",
        "grad ms",
        "v ns/NM",
        "g ns/NM",
        "fit s",
        "iter",
        "f",
        "g"
    )
    for case in cases, dtype in (Float64, Float32)
        site_speed(case, dtype)
    end
    if "threads" in args
        @printf("\n%-12s %11s %11s %11s\n", "case", "", "serial", "threaded")
        foreach(throughput, cases)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
