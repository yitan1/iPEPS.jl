using Test
using iPEPS
using iPEPS: honeycomb, init_hb_gs, check_Q_op, get_ghz_111, get_ghz, get_Q_op, get_Q_ghz
using iPEPS: sigmax, sigmay, sigmaz
using OMEinsum
using TOML

@testset "honeycomb" begin
    @testset "D = 2, init gs" begin
        A = init_hb_gs(2, p1 = 0.24, p2 = 0, dir ="XX")
        H = honeycomb(1, 1, dir = "XX")
        cfg = TOML.parsefile("src/default_config.toml")
        e0, _ = compute_gs_energy(A, H, cfg)

        @test isapprox(real(e0)/8, -0.16349, atol = 1e-5)
    end

    @testset "D = 4, init gs" begin
        A = init_hb_gs(4, p1 = 0.24, p2 = 0)
        H = honeycomb(1, 1)
        cfg = TOML.parsefile("src/default_config.toml")
        e0, _ = compute_gs_energy(A, H, cfg)

        @test isapprox(real(e0)/8, -0.19643, atol = 1e-5)
    end
end

@testset "check_Q_op" begin
    ans, Q_op = check_Q_op()
    @test ans[1]
    @test ans[2]
    @test ans[3]

    @testset "sigma * q" begin
        Q_op = get_Q_op()

        v = [0.0 im; 1 0]
        v = (sigmax - sigmay)/sqrt(2)#*exp(im*pi/4)
        vt = [0.0 1; im 0]
        vvt = sigmaz

        # sigma * Q
        @ein Qx[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, p1, m5] * sigmax[m4, p1] 
        @ein Qy[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, p1, m5] * sigmay[m4, p1] 
        @ein Qz[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, p1, m5] * sigmaz[m4, p1]
    
        @ein Qx2[m1, m2, m3, m4, m5] := v[m2, p1] * Q_op[m1, p1, p2, m4, m5] * (v')[p2, m3]
        @ein Qy2[m1, m2, m3, m4, m5] := v[m3, p2] * Q_op[p1, m2, p2, m4, m5] * (v')[p1, m1]
        @ein Qz2[m1, m2, m3, m4, m5] := v[m1, p1] * Q_op[p1, p2, m3, m4, m5] * (v')[p2, m2]

        @test approx(sum(Qx .- Qx2), 0.0, atol = 1e-12)
        @test approx(sum(Qy .- Qy2), 0.0, atol = 1e-12)
        @test approx(sum(Qz .- Qz2), 0.0, atol = 1e-12)
    end

    @test "Z2 Q_op" begin
        Q_op = get_Q_op()
        # g * Q * g * g
        @ein Qg[m1, m2, m3, m4, m5] := Q_op[p1, p2, p3, m4, m5] * sigmaz[p1, m1] * sigmaz[p2, m2] * sigmaz[p3, m3]

        @test Qg == Q_op
    end

    # n*Q
    # n_sigma = 1 / sqrt(3) * iPEPS.sigmax + 1 / sqrt(3) * iPEPS.sigmay + 1 / sqrt(3) * iPEPS.sigmaz
    # @ein Qxyz[m1, m2, m3, m4, m5] := sigmaz[m4, p1] * Q_op[m1, m2, m3, p1, p2] * sigmaz[p2, m5]
    # @ein Qxyz2[m1, m2, m3, m4, m5] := sigmaz[m3, p1] * Q_op[m1, m2, p1, m4, m5]
    # @test Qxyz == Qxyz2
end

@testset "check_GHZ" begin
    @testset "check_energy" begin
        ux, uy, uz = 1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)
        s111 = 1 / sqrt(2 + 2 * uz) * [1 + uz, ux + im * uy]
        uxm, uym, uzm = -1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)
        sm111 = 1 / sqrt(2 + 2 * uzm) * [1 + uzm, uxm + im * uym]

        A = get_Q_ghz()
        H = honeycomb(1, 1)
        cfg = TOML.parsefile("src/default_config.toml")
        e0, _ = compute_gs_energy(A, H, cfg)

        @test isapprox(real(e0)/8, -0.16349, atol = 1e-5)

        @test isapprox(sum(ghz[1,1,1,:]./exp(-im * pi/8) .- s111), 0.0, atol = 1e-5)
        @test isapprox(sum(ghz[2,2,2,:]./(-exp(-im * pi/8)) .- sm111), 0.0, atol = 1e-5)
    end

    @testset "check symmetry" begin
        ghz = get_ghz()
        op = sigmaz

        @ein g1[m1, m2, m3, m4] := ghz[p1, p2, p3, p4] * op[p1, m1] * op[p2, m2] * op[p3, m3] * op[p4, m4]

        @test g1 == ghz
    end

    # @testset "check_111_symmetry" begin
    #     op = iPEPS.sigmaz
    #     ghz = get_ghz_111()
    #     @ein g1[m1, m2, m3, m4] := ghz[p1, p2, p3, p4] * op[p1, m1] * op[p2, m2] * op[p3, m3] * op4[p4, m4]

    #     @test g1 == ghz
    # end
end


ghz = get_ghz()
phi = pi / 4
cost2 = sqrt((1+1/sqrt(3))/2)
sint2 = sqrt((1-1/sqrt(3))/2)
S = [exp(-im * phi/2) * cost2 -exp(-im * phi/2) * sint2; exp(im * phi/2) * sint2 exp(im * phi/2) * cost2] 
op1 = sigmax
op2 = sigmay
op3 = sigmaz

@ein g1[m1, m2, m3, m4] := ghz[p1, p2, p3, p4] * op1[p1, m1] * op1[p2, m2] * op1[p3, m3] * op1[p4, m4]

@ein g2[m1, m2, m3, m4] := ghz[p1, p2, p3, p4] * op2[p1, m1] * op2[p2, m2] * op2[p3, m3] * op2[p4, m4]

@ein g3[m1, m2, m3, m4] := ghz[m1, p2, p1, m4] * op3[p1, m3] * op3[p2, m2]

g1 +g2 

ghz