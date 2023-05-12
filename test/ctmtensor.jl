using iPEPS
using BenchmarkTools
using LinearAlgebra

using Accessors

D = 2
d = 4
A = randn(D,D,D,D,d)
A = A/maximum(abs,A);
ts0 = iPEPS.CTMTensors(A,A);
@code_warntype iPEPS.right_rg(ts0, 50)

function test_ad(A)
    A = A./ norm(A)
    A[1]
end
@time test_ad(A) 

using Zygote
@time y, dx = pullback(test_ad,  A);
@time dx(1);
@btime gradient(test_ad, A);


