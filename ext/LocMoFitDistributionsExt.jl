module LocMoFitDistributionsExt

using LocMoFit
using Random
using Distributions

function repeat_cols(A::AbstractMatrix, B::AbstractVector{<:Integer})
    m, n = size(A)
    out = similar(A, m, sum(B))
    k = 1
    @inbounds for j in 1:n
        bj = B[j]
        out[:, k:k+bj-1] .= A[:, j]
        k += bj
    end
    return out
end

function sample_labeled_pts(rng::AbstractRNG, pts::AbstractMatrix{<:Real}, labeling_efficiency::Real)
    mask = rand(rng, Bernoulli(labeling_efficiency), size(pts, 2))
    return pts[:, mask]
end

sample_nevents(rng::AbstractRNG, mean_events_per_dye::Real, npoints::Int) = rand(rng, Poisson(mean_events_per_dye), npoints)

sample_nphotons(rng::AbstractRNG, mean_photons_per_event::Real, nevents::AbstractVector{<:Integer}) = reduce(vcat, rand.(rng, Poisson(mean_photons_per_event), nevents))

function generate_data(rng::AbstractRNG, pts::AbstractMatrix{<:Real}; psf_σ::Real, mean_photons_per_event::Real, mean_events_per_dye::Real, labeling_efficiency::Real, z_factor::Real=2.5)
    labeled_pts = sample_labeled_pts(rng, pts, labeling_efficiency)
    nevents = sample_nevents(rng, mean_events_per_dye, size(labeled_pts, 2))
    locs = repeat_cols(labeled_pts, nevents)
    nphotons = sample_nphotons(rng, mean_photons_per_event, nevents)

    mask = nphotons .> 0
    locs, nphotons = locs[:, mask], nphotons[mask]

    loc_precs = psf_σ ./ nphotons'
    spreads = randn!(rng, similar(locs)) .* loc_precs
    size(locs, 1) == 3 && (spreads[3, :] .*= z_factor)

    locs .+= spreads
    return locs, loc_precs
end

function generate_data2(rng::AbstractRNG, pts::AbstractMatrix{<:Real}; psf_σ::Real, mean_photons_per_event::Real, mean_events_per_dye::Real, labeling_efficiency::Real, z_factor::Real=2.5)
    ndims = size(pts, 1)
    locs = similar(pts, ndims, 0)
    loc_precs = similar(pts, ndims, 0)

    labeled_pts = sample_labeled_pts(rng, pts, labeling_efficiency)

    for pt in eachcol(labeled_pts)
        nevents = rand(rng, Poisson(mean_events_per_dye))
        nevents == 0 && continue
        nphotons = rand(rng, Poisson(mean_photons_per_event), nevents)
        mask = nphotons .> 0
        nphotons = nphotons[mask]
        nevents -= count(!, mask)
        loc_prec = psf_σ ./ nphotons'
        loc_prec_mat = ndims == 3 ? [1, 1, 2.5] .* loc_prec : repeat(loc_prec, 2)
        pts_loc = loc_prec_mat .* randn(rng, ndims, nevents) .+ pt
        locs = hcat(locs, pts_loc)
        loc_precs = hcat(loc_precs, loc_prec_mat)
    end
    return locs, loc_precs
end

end