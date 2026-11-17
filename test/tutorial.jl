# Helpers shared by test/fit.jl and the benchmarks (benchmarks/accuracy/fit.jl, benchmarks/speed/speed.jl): reading the CSVs DynMoSim writes, the tutorial sites (examples/data), the moved start, and comparing axes and structures. Needs only LocMoFit and LinearAlgebra.
using LocMoFit, LinearAlgebra
using LocMoFit: fit_problem, full_parameters

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

# the position and precision columns of D-dimensional data
function columns(D)
    return if D == 3
        ((:x_nm, :y_nm, :z_nm), (:x_err_nm, :y_err_nm, :z_err_nm))
    else
        ((:x_nm, :y_nm), (:x_err_nm, :y_err_nm))
    end
end
# the columns `cols` of the table `d`, for the rows `rows`, as a D×N matrix
function matrix(d, cols, rows=eachindex(d[cols[1]]))
    return permutedims(reduce(hcat, [d[c][rows] for c in cols]))
end

const TUTORIAL = joinpath(@__DIR__, "..", "examples", "data")
# the true parameters of a tutorial site
function tutorial_parameters(case)
    return map(first, readcsv(joinpath(TUTORIAL, "$(case)_parameters.csv")))
end
# a tutorial site: its localizations and precisions, its true parameters, which localizations are signal, and their components
function tutorial(case, D)
    d = readcsv(joinpath(TUTORIAL, "$case.csv"))
    t = readcsv(joinpath(TUTORIAL, "$(case)_truth.csv"))
    cols = columns(D)
    locs, precs = matrix(d, cols[1]), matrix(d, cols[2])
    return (locs, precs, tutorial_parameters(case), t.is_signal .> 0.5, t.component)
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

"""
    start_params(θ; fixed=(), priors=(;))

Starting values given in LocMoFit's parameter names (`theta`, `phi` for an axis, `alpha` for a cap's half-angle, `omega0`, `omega2` for the fractions), or in those of the tutorial's parameter files, which use earlier parametrizations: a cap's curvature `eta` (κ′ = tanh(`eta`)) and the signal fraction `logit_prob` (π = logistic(`logit_prob`)), as the `params` of `run_locmofit`: each quantity's `ParamToFit` with its value (an axis as `(theta, phi)`, κ′ = sin(`alpha`/2) or tanh(`eta`), the fractions softmax(`omega0`, 0, `omega2`) or, with `logit_prob`, (1 − π, π·softmax(0, `omega2`))), held (`Fixed()`) if its name is in `fixed`, and with its prior from `priors` if named there (else its default).
"""
function start_params(θ; fixed=(), priors=(;))
    values = Pair{Symbol,Any}[]
    for (k, v) in pairs(θ)
        name = string(k)
        if startswith(name, "theta")
            suffix = name[6:end]
            push!(values, Symbol(:axis, suffix) => (v, θ[Symbol(:phi, suffix)]))
        elseif startswith(name, "alpha")
            push!(values, Symbol(:kappa, name[6:end]) => sin(v / 2))
        elseif startswith(name, "eta")
            push!(values, Symbol(:kappa, name[4:end]) => tanh(v))
        elseif !startswith(name, "phi") && !(k in (:omega0, :omega2, :logit_prob))
            push!(values, k => v)
        end
    end
    lw = haskey(θ, :omega2) ? [0.0, θ.omega2] : [0.0]
    w = softmax(lw)
    if haskey(θ, :omega0)
        push!(values, :fractions => Tuple(softmax([θ.omega0; lw])))
    elseif haskey(θ, :logit_prob)
        signal = 1 / (1 + exp(-θ.logit_prob))
        push!(values, :fractions => (1 - signal, (signal .* w)...))
    elseif haskey(θ, :omega2)
        push!(values, :fractions => Tuple(w))
    end
    return NamedTuple(
        k => ParamToFit(v, k in fixed ? Fixed() : get(priors, k, nothing)) for
        (k, v) in values
    )
end

softmax(x) = exp.(x) ./ sum(exp.(x))

angle_deg(a, b) = acosd(clamp(dot(a, b) / (norm(a) * norm(b)), -1, 1))   # a cap's axis has a sign
axis_deg(a, b) = min(angle_deg(a, b), 180 - angle_deg(a, b))             # a ring's axis has none
mirror(a) = [a[1], a[2], -a[3]]

# the true structure in the world frame, as run_locmofit reports a fitted one: started at the truth (the starting axes at the true ones, the chart coordinates at 0), the starting point is the truth
function true_components(model, truth, locs, precs)
    p, u = fit_problem(LocMoFit.build_model(model), locs, precs; params=start_params(truth))
    return LocMoFit.components(p.model, p.unit_model, full_parameters(u, p.parameter_map))
end
