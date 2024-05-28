using iPEPS
using Optim
using LinearAlgebra
using Zygote
using Random, TOML
using Test

@testset "D = 2, init gs" begin
    A = init_hb_gs(2, p1 = 0.24, p2 = 0, dir ="XX")
    H = honeycomb(1, 1, dir = "XX")
    cfg = TOML.parsefile("src/optimize/default_config.toml")
    e0, _ = compute_gs_energy(A, H, cfg)

    @test isapprox(real(e0)/8, -0.16349, atol = 1e-5)
end

D = 2
d = 2
rng = MersenneTwister(3)
A = randn(rng, ComplexF64, D,D,D,D,d);

H = ising();

optim_gs(H, A, "")






