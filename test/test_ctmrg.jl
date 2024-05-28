using iPEPS
using TOML
using Test

@testset "D = 2, init gs" begin
    A = init_hb_gs(2, p1 = 0.24, p2 = 0, dir ="XX")
    H = honeycomb(1, 1, dir = "XX")
    cfg = TOML.parsefile("src/optimize/default_config.toml")
    e0, _ = compute_gs_energy(A, H, cfg)

    @test isapprox(real(e0)/8, -0.16349, atol = 1e-5)
end


H = ising();
cfg = TOML.parsefile("src/optimize/default_config.toml");

prepare_basis(H, cfg);

# optim_es(0, 0, cfg)