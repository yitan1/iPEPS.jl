
using iPEPS
# using MKL

H = ising()

A = randn(2,2,2,2,2);

res = optim_gs(H, A, "")

optim_es(H, 0., 0., "")   


e, v = iPEPS.evaluate_es(0,0) 

