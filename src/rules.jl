# Hand-written Enzyme reverse rule for log_likelihood, the loop over the localizations and the model points in `loss`. Differentiating that loop would cost Enzyme a tape of every term; with the rule, Enzyme differentiates only what lies around it (the model points from the parameters, the weights, the background) and takes the loop's gradient from likelihood_pass, which computes it along with the value. So any change to the likelihood in likelihood_pass must be made in its gradient as well. The localizations, their precisions and the ranges are treated as constants.

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(log_likelihood)},
    ::Type{<:Active},
    args::Vararg{Annotation,7},
)
    ℓ, ∇ℓ = likelihood_pass(map(a -> a.val, args)..., Val(true))
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? ℓ : nothing, nothing, ∇ℓ
    )
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(log_likelihood)},
    dret::Active,
    ∇ℓ,
    locs,
    loc_prec,
    points,
    ranges,
    σ,
    logw,
    b,
)
    # with runtime activity, inactive points may come as Duplicated with the primal as their shadow
    points isa Const ||
        points.dval === points.val ||
        (points.dval .+= dret.val .* ∇ℓ.points)
    active(arg, g) = arg isa Active ? dret.val .* g : nothing
    return (
        nothing,
        nothing,
        nothing,
        nothing,
        active(σ, ∇ℓ.σ),
        active(logw, ∇ℓ.logw),
        active(b, ∇ℓ.b),
    )
end
