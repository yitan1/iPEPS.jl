
using iPEPS
# using MKL

H = ising()

A = randn(2,2,2,2,2);

res = optim_gs(H, A)

using JLD2 

A = load("simulation/ising_default_D2_X50/gs.jld2", "A"); 
optim_es(A, H, 0., 0.)   


e, v = iPEPS.evaluate_es(0,0)