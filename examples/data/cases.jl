# The simulated sites behind LocMoFit.jl's tutorial data (examples/data/generate.jl) and its accuracy benchmark (benchmarks/accuracy/generate.jl): for each model, the true parameters in LocMoFit's parametrization, the same structure built for DynMoSim, and the acquisition. Lengths in nm, angles in radians. Each axis is given as θ, ϕ (polar angle and azimuth; a composite's second axis in the first one's frame), which is how LocMoFit takes a starting axis. Needs DynMoSim (and StaticArrays).
using DynMoSim, StaticArrays, LinearAlgebra, Random

direction(θ, ϕ) = SA[sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
# the minimal rotation taking z to direction(θ, ϕ), which is LocMoFit's rotmat(θ, ϕ)
rot(θ, ϕ) = θ == 0 ? SMatrix{3,3,Float64}(I) : rotation(direction(θ, ϕ))

# DynMoSim's constant-area cap with scale s and dimensionless curvature κ is LocMoFit's cap with scale s and κ′ = tanh(eta)
function cap(apex, R, s, κ)
    return Structure(SphereSurface(), Similarity(; scale=s, rot=R, shift=apex) ∘ Bend(κ))
end
ring(center, R, r) = Structure(Ring(), Similarity(; scale=r, rot=R, shift=center))

ring_structure(p) = ring(SA[p.x, p.y, p.z], rot(p.theta, p.phi), p.r)
spcap_structure(p) = cap(SA[p.x, p.y, p.z], rot(p.theta, p.phi), p.s, tanh(p.eta))
# the components' shares of the signal localizations, from LocMoFit's omega2 = log(w₂/w₁)
shares(p) = (1 / (1 + exp(p.omega2)), 1 / (1 + exp(-p.omega2)))
# ring 2 is placed in ring 1's frame
function double_ring_structure(p)
    R1, t1 = rot(p.theta1, p.phi1), SA[p.x1, p.y1, p.z1]
    w = shares(p)
    return Composite(
        ring(t1, R1, p.r1) => w[1],
        ring(t1 + R1 * SA[p.x2, p.y2, p.z2], R1 * rot(p.theta2, p.phi2), p.r2) => w[2],
    )
end
# the cap is placed in the ring's frame, by its apex
function spcap_ring_structure(p)
    R1, t1 = rot(p.theta1, p.phi1), SA[p.x1, p.y1, p.z1]
    w = shares(p)
    return Composite(
        ring(t1, R1, p.r1) => w[1],
        cap(
            t1 + R1 * SA[p.x2, p.y2, p.z2], R1 * rot(p.theta2, p.phi2), p.s2, tanh(p.eta2)
        ) => w[2],
    )
end

# a uniform background makes up 15% of the localizations (where there is one), so the true signal fraction is 0.85
const BACKGROUND_FRACTION = 0.15
const LOGIT_SIGNAL = log((1 - BACKGROUND_FRACTION) / BACKGROUND_FRACTION)

# model name, true parameters (with the signal fraction, logit_prob, and the composites' shares, omega2: equal for the two rings, 30% ring and 70% cap), structure, and the mean number of signal localizations per site
const CASES = (
    ring=(
        model="Ring",
        truth=(
            x=10.0,
            y=-5.0,
            z=20.0,
            r=50.0,
            theta=0.3,
            phi=0.8,
            sigma=0.0,
            logit_prob=LOGIT_SIGNAL,
        ),
        structure=ring_structure,
        nsignal=100,
    ),
    spcap=(
        model="SphericalCap",
        truth=(
            x=5.0,
            y=10.0,
            z=40.0,
            s=60.0,
            eta=atanh(0.5),
            theta=0.25,
            phi=-1.0,
            sigma=0.0,
            logit_prob=LOGIT_SIGNAL,
        ),
        structure=spcap_structure,
        nsignal=250,
    ),
    double_ring=(
        model="DoubleRing",
        truth=(
            x1=0.0,
            y1=0.0,
            z1=-25.0,
            r1=50.0,
            theta1=0.2,
            phi1=0.5,
            x2=0.0,
            y2=0.0,
            z2=50.0,
            r2=50.0,
            theta2=0.0,
            phi2=0.0,
            sigma=0.0,
            omega2=0.0,
            logit_prob=LOGIT_SIGNAL,
        ),
        structure=double_ring_structure,
        nsignal=160,
    ),
    spcap_ring=(
        model="SpCapRing",
        truth=(
            x1=0.0,
            y1=0.0,
            z1=-30.0,
            r1=50.0,
            theta1=0.2,
            phi1=0.5,
            x2=0.0,
            y2=0.0,
            z2=90.0,
            s2=25.0,
            eta2=atanh(0.5),
            theta2=0.0,
            phi2=0.0,
            sigma=0.0,
            omega2=log(7 / 3),
            logit_prob=LOGIT_SIGNAL,
        ),
        structure=spcap_ring_structure,
        nsignal=300,
    ),
)

# acquisition: shot-noise-limited precision (PSF σ 150 nm laterally, 2.5× axially; 500 photons per event on average), and the background filling the site's field of view, a 300 nm cube (SMAP's default ROI size)
const OPTICS = Optics(; uncertainty=ShotNoise(150.0), brightness=5e4, ontime=0.01)
const FOV = FieldOfView(SA[-150.0, -150.0, -150.0], SA[150.0, 150.0, 150.0])

# one site of `case`, over one second of acquisition (so the rates are mean counts): DynMoSim's localizations and truth
function simulate_site(rng, case; background=true)
    bg = if background
        Background(;
            rate=case.nsignal * BACKGROUND_FRACTION / (1 - BACKGROUND_FRACTION),
            fov=FOV,
            meanphotons=300.0,
        )
    else
        nothing
    end
    return simulate(
        rng,
        case.structure(case.truth);
        emis_rate=case.nsignal,
        t_end=1.0,
        optics=OPTICS,
        background=bg,
    )
end

function writecsv(path, table)
    open(path, "w") do io
        println(io, join(keys(table), ","))
        for i in eachindex(first(table))
            println(io, join((string(col[i]) for col in table), ","))
        end
    end
end
