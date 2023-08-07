using iPEPS
using MKL
# using Optim
using LinearAlgebra
using Zygote
using NPZ
using ProfileView 
# ProfileView.@profview profile_test(10)

H = iPEPS.honeycomb(1, 1);

A = iPEPS.init_hb_gs() |> real;

iPEPS.compute_gs_energy(A, H)

A - permutedims(conj(A), (3,2,1,4,5)) 

D = 2
d = 4
A = randn(D,D,D,D,d);
A = A ./ maximum(abs,A);

A = npzread("honey_015_A.npz")["A"] 
A = dropdims(A, dims = 1)

ts0 = iPEPS.CTMTensors(A,A);
conv_fun(_x) = iPEPS.get_gs_energy(_x, H)
@time ts3, s = iPEPS.run_ctm(ts0, 50, conv_fun = conv_fun);
iPEPS.run_energy(H, ts0, 30, A)

gradient(x -> iPEPS.run_energy(H, ts0, 50, x), A)

iPEPS.optim_GS(H, A)
 

 