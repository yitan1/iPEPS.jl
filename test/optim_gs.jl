using iPEPS
# using Optim
using LinearAlgebra
using Zygote
using ProfileView 
# ProfileView.@profview profile_test(10)

using NPZ
function npz2jld(A)
    # A = npzread("one_h3_X96.npz")["A"] 
    A = permutedims(A,(5,4,3,2,1))
    A = permutedims(A,(3,4,1,2,5))
    cfg = TOML.parsefile("src/default_config.toml");
    iPEPS.compute_gs_energy(A, H, cfg)
    jldsave("gs.jld2", A = A)
end


H = iPEPS.honeycomb(1, 1);

A = iPEPS.init_hb_gs(2);

iPEPS.compute_gs_energy(A, H)

A - permutedims(conj(A), (3,2,1,4,5)) 

D = 2
d = 4
A = randn(D,D,D,D,d);
A = A ./ maximum(abs,A);

ts0 = iPEPS.CTMTensors(A,A);
conv_fun(_x) = iPEPS.get_gs_energy(_x, H)
@time ts3, s = iPEPS.run_ctm(ts0, 50, conv_fun = conv_fun);

iPEPS.run_energy(H, ts0, 30, A)

gradient(x -> iPEPS.run_energy(H, ts0, 50, x), A)

iPEPS.optim_GS(H, A)



 

 