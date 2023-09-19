using iPEPS
using JLD2
# using MKL
# using Random  

# rng = MersenneTwister(4);
# D = 2
# A = randn(rng, Float64, D,D,D,D,4);
# A = iPEPS.get_symmetry(A)

A = iPEPS.init_hb_gs(4, p1 = 0.24, p2 = 0);
jldsave("simulation/hb_g11_D4_X64/gs.jld2", A = A)
H = iPEPS.honeycomb(1, 1);
# res = optim_gs(H .*2, A, "")

prepare_basis(H, "")

pxs, pys = make_es_path()
n = 20
px, py = pxs[n], pys[n]
optim_es(px, py, "");
