using iPEPS
using Test
using TOML, Random

@testset "test_h4" begin
    h = iPEPS.hb_h4_ZZ()
    s1, s2, s3, s4 = get_local_h(h)

    A0 = init_hb_gs(2, p1=0.24, p2=0, dir="ZZ")
    cfg = TOML.parsefile("src/optimize/default_config.toml")
    ts0 = iPEPS.CTMTensors(A0, cfg)
    ts, _ = iPEPS.run_ctm(ts0)

    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es
    A = ts.A
    A1, A2, A3, A4 = A, A, A, A
    A1d, A2d, A3d, A4d = ts.Ad, ts.Ad, ts.Ad, ts.Ad

    op_A1d, op_A2d, op_A3d, op_A4d = get_op_Ad4(s1, s2, s3, s4, A1d, A2d, A3d, A4d)

    energy, norm = iPEPS.energy_norm_4x4(C1, C2, C3, C4, E1, E2, E3, E4, A1, A2, A3, A4, op_A1d, op_A2d, op_A3d, op_A4d)
    e0 = energy[] / norm[]
    @show energy, norm, e0

    @test isapprox(real(e0) / 8, -0.16349, atol=1e-5)
end

D = 2
d = 2
rng = MersenneTwister(3)
A = randn(rng, ComplexF64, D, D, D, D, d)

H = ising_h4(2.5)
cfg = TOML.parsefile("src/optimize/default_config.toml")
optim_gs_h4(H, A, cfg)

prepare_basis_h4(H, cfg)

px, py = make_es_path()
for i in eachindex(px)
    optim_es_h4(px[i], py[i], cfg)
end

E = plot_band(4, cfg)

# using Plots
# plot(E')