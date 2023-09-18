using Test
using iPEPS
using iPEPS: honeycomb, init_hb_gs, check_Q_op
using TOML

@testset "honeycomb" begin
    @testset "D = 2, init gs" begin
        A = init_hb_gs(2, p1 = 0.24, p2 = 0)
        H = honeycomb(1, 1)
        cfg = TOML.parsefile("src/default_config.toml")
        e0, _ = compute_gs_energy(A, H, cfg)

        @test isapprox(real(e0)/4, -0.16348, atol = 1e-5)
    end

    @testset "D = 4, init gs" begin
        A = init_hb_gs(4, p1 = 0.24, p2 = 0)
        H = honeycomb(1, 1)
        cfg = TOML.parsefile("src/default_config.toml")
        e0, _ = compute_gs_energy(A, H, cfg)

        @test isapprox(real(e0)/4, -0.19643, atol = 1e-5)
    end

    @testset "check_Q_op" begin
        ans = check_Q_op()
        @test ans[1]
        @test ans[2]
        @test ans[3]
    end
end
