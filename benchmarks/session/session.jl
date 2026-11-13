# What a fresh Julia session pays before LocMoFit.jl fits fast, as it would in SMAPpy through juliacall (NOTES.md, Using LocMoFit.jl from SMAPpy): the time to load the package, the first fit of every model on its tutorial site (examples/data), which compiles it, the median of five fits after it, and the peak memory. The counterpart for LocMoFitPy is its benchmarks/speed/session.py. The script must be the session's first work, so run it in a new process, from the LocMoFit.jl folder:
#     julia --project=. benchmarks/session/session.jl
t0 = time()
using LocMoFit
const t_load = time() - t0
include(joinpath(@__DIR__, "..", "..", "test", "tutorial.jl"))
using Printf, Statistics

const CASES = (
    (name="ring", model="Ring", D=3, dtype=Float64),
    (name="spcap", model="SphericalCap", D=3, dtype=Float64),
    (name="double_ring", model="DoubleRing", D=3, dtype=Float64),
    (name="spcap_ring", model="SpCapRing", D=3, dtype=Float64),
    (name="ring2D", model="Ring", D=2, dtype=Float64),
    (name="ring", model="Ring", D=3, dtype=Float32),
)

rss() = Sys.maxrss() / 2^20

println("Julia $VERSION, $(Sys.cpu_info()[1].model), $(Threads.nthreads()) thread(s)")
@printf("load: %.1f s, peak memory %.0f MB\n", t_load, rss())
for c in CASES
    locs, precs = tutorial(c.name, c.D)
    t1 = @elapsed run_locmofit(c.model, locs, precs; dtype=c.dtype)
    t2 = median([@elapsed(run_locmofit(c.model, locs, precs; dtype=c.dtype)) for _ in 1:5])
    @printf(
        "%-26s first fit %5.1f s, then %6.1f ms, peak memory %.0f MB\n",
        "$(c.model) $(c.D)D $(c.dtype)",
        t1,
        1e3t2,
        rss()
    )
    flush(stdout)
end
@printf("total since start: %.1f s\n", time() - t0)
