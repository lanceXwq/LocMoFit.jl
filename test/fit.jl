# Fits of the tutorial data (examples/data: one simulated site per model, 15% background, a 300 nm field of view), from the benchmark's moved start: positions +5 nm, sizes ×1.15 (a cap's κ′ ×0.85), starting axes moved by 0.15 rad in θ and ϕ, blur 2 nm, the default signal fraction and shares. Every error must stay within 4× the scatter of one fit, measured over the 20 sites of the accuracy benchmark in the same setting (benchmarks/accuracy/share_prior_2026-11-14.txt, background, moved start, Float64; the simple models' values are those of prior_5_2_2026-10-28.txt, which the share prior does not change); an axis's error within its mean plus 4× its scatter. Under the default prior Beta(5, 2) the signal fraction showed no bias there, so it gets 4× its scatter too. The default share prior moves the ring's share of the cap with a ring by +0.0075 there (from 0.3 toward 1/2, derivations/derivation.qmd, @eq-share-map), so that share must stay within 4× its scatter of that bias.

include(joinpath(@__DIR__, "tutorial.jl"))

@testset "fit of the ring" begin
    locs, precs, truth, signal = tutorial("ring", 3)
    r = run_locmofit("Ring", locs, precs; init_params=moved(truth), fov_volume=300.0^3)
    c, t = only(r.components), only(true_components("Ring", truth, locs, precs))
    @test string(r.result.retcode) == "Success"
    @test all(abs.(c.position - t.position) .< 4 .* [1.254, 1.055, 1.351])
    @test abs(c.r - t.r) < 4 * 0.5465
    @test axis_deg(c.axis, t.axis) < 3.053 + 4 * 1.833
    @test abs(r.signal_fraction - mean(signal)) < 4 * 0.01412
end

@testset "fit of the ring in 2D" begin
    # z is undetermined in 2D and frozen; the axis is known only up to its mirror image through the xy plane
    locs, precs, truth, signal = tutorial("ring2D", 2)
    r = run_locmofit(
        "Ring",
        locs,
        precs;
        init_params=merge(moved(truth), (z=truth.z,)),
        freeze=(:z,),
        fov_volume=300.0^2,
    )
    c, t = only(r.components), only(true_components("Ring", truth, locs, precs))
    @test string(r.result.retcode) == "Success"
    @test all(abs.(c.position[1:2] - t.position[1:2]) .< 4 .* [1.321, 1.136])
    @test abs(c.r - t.r) < 4 * 0.7497
    @test min(axis_deg(c.axis, t.axis), axis_deg(c.axis, mirror(t.axis))) <
        8.676 + 4 * 4.958
    @test abs(r.signal_fraction - mean(signal)) < 4 * 0.01643
end

@testset "fit of the double ring" begin
    locs, precs, truth, signal, component = tutorial("double_ring", 3)
    r = run_locmofit(
        "DoubleRing", locs, precs; init_params=moved(truth), fov_volume=300.0^3
    )
    c, t = r.components, true_components("DoubleRing", truth, locs, precs)
    @test string(r.result.retcode) == "Success"
    @test all(abs.(c[1].position - t[1].position) .< 4 .* [0.9888, 1.065, 2.576])
    @test all(abs.(c[2].position - t[2].position) .< 4 .* [1.221, 1.387, 2.473])
    @test abs(c[1].r - t[1].r) < 4 * 1.002 && abs(c[2].r - t[2].r) < 4 * 0.6958
    @test axis_deg(c[1].axis, t[1].axis) < 3.837 + 4 * 2.595
    @test axis_deg(c[2].axis, t[2].axis) < 4.687 + 4 * 2.659
    @test abs(r.signal_fraction - mean(signal)) < 4 * 0.01164
    @test abs(r.weights[1] - mean(component[signal] .== 1)) < 4 * 0.03137
end

@testset "fit of the spherical cap" begin
    locs, precs, truth, signal = tutorial("spcap", 3)
    r = run_locmofit(
        "SphericalCap", locs, precs; init_params=moved(truth), fov_volume=300.0^3
    )
    c, t = only(r.components), only(true_components("SphericalCap", truth, locs, precs))
    @test string(r.result.retcode) == "Success"
    @test all(abs.(c.position - t.position) .< 4 .* [3.004, 3.36, 1.735])
    @test abs(c.curvature - t.curvature) < 4 * 2.009e-4
    @test abs(c.alpha - t.alpha) < 4 * 0.03445
    @test angle_deg(c.axis, t.axis) < 2.177 + 4 * 1.428
    @test abs(r.signal_fraction - mean(signal)) < 4 * 0.01511
end

@testset "fit of the cap with a ring" begin
    locs, precs, truth, signal, component = tutorial("spcap_ring", 3)
    r = run_locmofit("SpCapRing", locs, precs; init_params=moved(truth), fov_volume=300.0^3)
    c, t = r.components, true_components("SpCapRing", truth, locs, precs)
    center(k) = k.position - k.radius * k.axis                           # the cap's sphere's center
    @test string(r.result.retcode) == "Success"
    @test all(abs.(c[1].position - t[1].position) .< 4 .* [1.281, 1.515, 2.285])
    @test abs(c[1].r - t[1].r) < 4 * 0.8052
    @test axis_deg(c[1].axis, t[1].axis) < 3.624 + 4 * 1.735
    @test all(abs.(center(c[2]) - center(t[2])) .< 4 .* [2.181, 2.418, 11.08])
    @test abs(c[2].radius - t[2].radius) < 4 * 8.979 &&
        abs(c[2].alpha - t[2].alpha) < 4 * 0.1926
    @test angle_deg(c[2].axis, t[2].axis) < 4.134 + 4 * 2.216
    @test abs(r.signal_fraction - mean(signal)) < 4 * 0.01061
    @test abs(r.weights[1] - mean(component[signal] .== 1) - 0.0075) < 4 * 0.01225
end

@testset "fits from the estimated start" begin
    # with init_params left out, every 3D tutorial fit starts from the data and reaches the optimum the truth-started fit reaches (or a better one): its loss no more than 0.01 per localization above, the threshold NOTES.md counts a miss by, while the wrong maxima found on simulated sites lay 0.04–0.7 above
    for (case, model) in (
        ("ring", "Ring"),
        ("spcap", "SphericalCap"),
        ("double_ring", "DoubleRing"),
        ("spcap_ring", "SpCapRing"),
    )
        locs, precs, truth, _ = tutorial(case, 3)
        a = run_locmofit(model, locs, precs; fov_volume=300.0^3)
        t = run_locmofit(model, locs, precs; init_params=truth, fov_volume=300.0^3)
        @test string(a.result.retcode) == "Success"
        @test a.loss < t.loss + 0.01
    end
end
