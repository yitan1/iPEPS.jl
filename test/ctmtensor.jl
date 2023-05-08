using iPEPS
using BenchmarkTools
time()

using Accessors

D = 4
d = 4
A = randn(D,D,D,D,d)
A = A/maximum(abs,A);
ts0 = iPEPS.CTMTensors(A,A);

function test_ad(A)
    # st = Base.time()
    ts = iPEPS.CTMTensors(A,A)
    P, Pd, s = iPEPS.get_projector_left(ts, 100);
    newC1, newE4, newC4 = iPEPS.proj_left(ts, P, Pd);
    newCs = [newC1, ts.Cs[2], ts.Cs[3], newC4]
    newEs = [ts.Es[1], ts.Es[2], ts.Es[3], newE4]
    # @reset ts.Es = newEs
    # @reset ts.Cs = newCs
    ts = setproperties(ts, Es = newEs, Cs = newCs)
    # ed = Base.time()
    # println("+++", ed -st)
    ts.Es[4][1]
    # newE4[1]
end
@btime test_ad(A) 

using Zygote
y, dx = pullback(test_ad,  A);
dx(1)[1] |> size 
@btime gradient(test_ad, A);

Base.time_ns()


