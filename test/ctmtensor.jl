using iPEPS
time()

using Accessors

D = 4
d = 4
A = randn(D,D,D,D,d)
A = A/maximum(abs,A);
ts = iPEPS.CTMTensors(A,A);
P, Pd, s = iPEPS.get_projector_left(ts, 100);
newC1, newE4, newC4 = iPEPS.proj_left(ts, P, Pd);

@Zygote.nograd time
function test_ad(A, B)
    @time ts = iPEPS.CTMTensors(A,A)
    st = Base.time()
    @time sleep(1)
    ed = Base.time()
    ts = @set ts.B = B
    println("+++", ed -st)
    x = 4*ts.B
    ts = @set ts.B = x
    ts.B
end
test_ad(A,3);

using Zygote
y, dx = pullback(x->test_ad(A, x), 3);
dx(1)
gradient(x->test_ad(A, x), 3)

Base.time_ns()