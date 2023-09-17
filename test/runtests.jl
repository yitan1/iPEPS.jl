using iPEPS
using Test

@testset "iPEPS.jl" begin
    @testset "print test" begin
        include("test_print.jl")
    end

    @testset "io test" begin
        include("test_io.jl")
    end

    @testset "tcon test" begin
        include("test_tcon.jl")
    end

    @testset "model test" begin
        include("test_model.jl")
    end
end

########################

D = 30
A, B, C = rand(D, D, D), rand(D, D, D), rand(D, D, D);

function f(A, B, C)
    @tensor D[m1, m2, m3] := A[m1, p1, p2] * B[p1, p3, m2] * C[p2, p3, m3]
    D[1]
end
@btime f(A, B, C);

function f0(A, B, C)
    @tensor D[m1, m2, m3, m4] := A[m1, p1, m2] * B[p1, m3, m4]
    @tensor E[m1, m2, m3] := D[m1, m2, p1, p2] * C[p1, p2, m3]
    return E[1]
end
@time f0(A, B, C);
function f1(A, B, C)
    @ein D[m1, m2, m3] := A[m1, p1, p2] * B[p1, p3, m2] * C[p2, p3, m3]
    D[1]
end
@time f1(A, B, C);

function f11(A, B, C)
    @ein D[m1, m2, m3, m4] := A[m1, p1, m2] * B[p1, m3, m4]
    @ein E[m1, m2, m3] := D[m1, m2, p1, p2] * C[p1, p2, m3]
    return E[1]
end
@time f11(A, B, C);

function f2(A, B, C)
    D = contract(A, B, "aib", "icd", "abcd")
    E = contract(D, C, "abij", "ijc", "abc")
    E[1]
end
@time f2(A, B, C);

@time gradient(x -> f2(x, B, C), A);

##############################

A, B, C = rand(5, 5, 5, 5, 5, 5), rand(5, 5, 5), rand(5, 5, 5);
function f(A, B, C)
    ex = Meta.parse("@tensor F[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e]")
    m = @__MODULE__
    @eval m $(macroexpand(TensorOperations, :($ex)))
    sum(F)
end
f(A, B, C);
gradient(x -> f(x, B, C), A)




