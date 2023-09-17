using Test
using iPEPS: tcon
using OMEinsum

@testset "tcon" begin
    A, B = rand(5, 5, 5, 5), rand(5, 5, 5)

    @ein r0[m1, m2, m3] := A[p1, p2, m1, m2] * B[p1, p2, m3] 

    xs = [A, B]
    ind_xs = [[1, 2, -1, -2], [1, 2, -3]]
    r1 = tcon(xs, ind_xs)

    @test r0 ≈ r1
end