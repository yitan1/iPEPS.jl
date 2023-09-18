using Test
using iPEPS: NestedTensor, tcon

@testset "EmptyT" begin
    
end

@testset "NestedTensor" begin
    A = [rand(5,5,5) for i = 1:4]
    B = [rand(5,5,5) for i = 1:4]

    A1 = iPEPS.NestedTensor(A)
    B1 = iPEPS.NestedTensor(B)

    @test eachindex(A1) == eachindex(A1.data)
    @test size(A1) == size(A1.data[1])
    @test size(A1, 1) == size(A1.data[1], 1)
    @test A1[1] == A1.data[1]

    input = [[-1,1,2], [1,2,-2]]
    C = tcon([A1, B1], input)

    @test C[1] == tcon([A[1], B[1]], input)
    @test C[2] == tcon([A[1], B[2]], input) + tcon([A[2], B[1]], input)
    @test C[3] == tcon([A[1], B[3]], input) + tcon([A[3], B[1]], input)
    @test C[4] == tcon([A[1], B[4]], input) + tcon([A[4], B[1]], input) + tcon([A[2], B[3]], input) + tcon([A[3], B[2]], input)

end