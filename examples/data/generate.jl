# Tutorial data for LocMoFit.jl: one simulated site per model (cases.jl), written with SimSMLM. Run from the LocMoFit.jl folder:
#     julia --project=../SimSMLM examples/data/generate.jl
# Writes, next to this script, for each case: <case>.csv (localizations, SMAPpy column names), <case>_truth.csv (the true position of every localization and whether it is signal, row-aligned) and <case>_parameters.csv (the true parameters, in LocMoFit's names). ring2D is the ring site seen in 2D: the same rows without their z columns.
include(joinpath(@__DIR__, "cases.jl"))

drop_z(table) = NamedTuple(k => v for (k, v) in pairs(table) if !(k in (:z_nm, :z_err_nm)))
for (i, (name, case)) in enumerate(pairs(CASES))
    locs, truth = simulate_site(Xoshiro(i), case)
    writecsv(joinpath(@__DIR__, "$name.csv"), locs)
    writecsv(joinpath(@__DIR__, "$(name)_truth.csv"), truth)
    writecsv(joinpath(@__DIR__, "$(name)_parameters.csv"), map(v -> [v], case.truth))
    if name == :ring
        writecsv(joinpath(@__DIR__, "ring2D.csv"), drop_z(locs))
        writecsv(joinpath(@__DIR__, "ring2D_truth.csv"), drop_z(truth))
        writecsv(joinpath(@__DIR__, "ring2D_parameters.csv"), map(v -> [v], case.truth))
    end
    println("$name: $(length(locs.x_nm)) localizations ($(count(truth.is_signal)) signal)")
end
