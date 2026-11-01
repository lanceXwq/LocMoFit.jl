# Tutorial data for LocMoFit.jl: one simulated site per model (cases.jl), written with DynMoSim. Run from the LocMoFit.jl folder:
#     julia --project=../DynMoSim examples/data/generate.jl
# Writes, next to this script, for each case: <case>.csv (localizations, SMAPpy column names), <case>_truth.csv (the true position of every localization and whether it is signal, row-aligned) and <case>_parameters.csv (the true parameters, in LocMoFit's names). ring2D is the ring site seen in 2D: the same rows projected by DynMoSim's `project`, which drops z.
include(joinpath(@__DIR__, "cases.jl"))

for (i, (name, case)) in enumerate(pairs(CASES))
    locs, truth = simulate_site(Xoshiro(i), case)
    writecsv(joinpath(@__DIR__, "$name.csv"), smaptable(locs))
    writecsv(joinpath(@__DIR__, "$(name)_truth.csv"), smaptable(truth))
    writecsv(joinpath(@__DIR__, "$(name)_parameters.csv"), map(v -> [v], case.truth))
    if name == :ring
        locs2, truth2 = project(locs, truth)
        writecsv(joinpath(@__DIR__, "ring2D.csv"), smaptable(locs2))
        writecsv(joinpath(@__DIR__, "ring2D_truth.csv"), smaptable(truth2))
        writecsv(joinpath(@__DIR__, "ring2D_parameters.csv"), map(v -> [v], case.truth))
    end
    println("$name: $(length(locs)) localizations ($(nsignal(truth)) signal)")
end
