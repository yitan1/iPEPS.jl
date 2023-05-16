using iPEPS
using iPEPS: tcon, NestedTensor
using Test

# @testset "iPEPS.jl" begin
#     # Write your tests here.
# end

using BenchmarkTools
using LinearAlgebra
using Zygote
using MKL
using TensorOperations, TensorRules
using OMEinsum
using Tullio
using BliContractor


using TOML
@__DIR__
@time cfg = TOML.parsefile("src/config.toml")

using JLD2
filename = "abc"*"hello"*".jld2"
a = $filename
a = ((1,2,3),  (2,-2))
b = pi/5
jldsave("test1_$(filename)_$b"; a)
load("test"*filename, "a")
A = rand(10,10);
u,s,v = svd(A)

similar([A,A])
#######################
D = 100
A = rand(D,D);
@btime tr(A);

A, B, C = rand(5,5,5,5), rand(5,5,5), rand(5,5,5);

function ft(A, B, C)
    @tensor res[m1, m2, m3] := A[p1,p2,m1,m2]*B[p1,p2,m3] + C[m1,m2,m3]
    # return nothing 
end
function ft1(A, B, C)
    xs = [A,B]
    ind_xs = [[1,2,-1,-2], [1,2,-3]]
    ind_y = [-1,-2,-3]
    res = tcon(xs, ind_xs, ind_y)
    res  .+= C
    return nothing
end
@time r0 = ft(A, B, C);
@time r1 = ft1(A, B, C);


########################

D = 30
A, B, C = rand(D,D,D), rand(D,D,D), rand(D,D,D);

function f(A, B, C)
    @tensor D[m1, m2, m3] := A[m1, p1, p2]*B[p1,p3,m2]*C[p2, p3, m3]
    D[1]
end
@btime f(A,B,C);

function f0(A,B,C)
    @tensor D[m1, m2, m3, m4] := A[m1, p1, m2]*B[p1,m3,m4]
    @tensor E[m1, m2, m3] := D[m1,m2, p1,p2]*C[p1, p2, m3]
    return E[1]
end
@time f0(A,B,C);
function f1(A, B, C)
    @ein D[m1, m2, m3] := A[m1, p1, p2]*B[p1,p3,m2]*C[p2, p3, m3]
    D[1]
end
@time f1(A,B,C);

function f11(A,B,C)
    @ein D[m1, m2, m3, m4] := A[m1, p1, m2]*B[p1,m3,m4]
    @ein E[m1, m2, m3] := D[m1,m2, p1,p2]*C[p1, p2, m3]
    return E[1]
end
@time f11(A,B,C);

function f2(A,B,C)
    D = contract(A, B, "aib", "icd", "abcd")
    E = contract(D, C, "abij", "ijc", "abc")
    E[1]
end
@time f2(A,B,C);

@time gradient(x -> f2(x,B,C), A);



##############################

@macroexpand @tensor out[-1,-2,-3,-4] := B[1, -2, -1]*C[1, -3, -4]
ncon()

Tullio.@tensor F[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] 
# + C[c,a,b]

A, B, C = rand(5,5,5,5,5,5), rand(5,5,5), rand(5,5,5);
function f(A,B,C)
    ex = Meta.parse("@tensor F[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e]")
    m = @__MODULE__
    @eval m $(macroexpand(TensorOperations, :($ex)))
    sum(F)
end
f(A,B,C);
gradient(x -> f(x, B, C), A)






