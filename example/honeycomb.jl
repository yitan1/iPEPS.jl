
using iPEPS
using MKL
using TOML

using Random  

rng = MersenneTwister(1);
D = 3
A = randn(rng, Float64, D,D,D,D,4);

A = iPEPS.init_hb_gs(2) .|> real
H = iPEPS.honeycomb(1, 1);
res = optim_gs(H, A, "")

prepare_basis(H, "")
optim_es(0., 0., "")

using CairoMakie
s1 = iPEPS.Sz
s2 = iPEPS.SI
# op = iPEPS.tout(s1, s2) + iPEPS.tout(s2, s1)
op = iPEPS.tout(s1, s2) + iPEPS.tout(s2, s1) +  iPEPS.tout(s1, s1);
es, swk0 = compute_spectral(op, .0, .0, "")
scatter(es,swk0)

x, y = plot_spectral(es, swk0, factor = 0.04)
lines(x, y)


A1 = res.minimizer ./ maximum(A1)
cfg = TOML.parsefile("src/default_config.toml");
iPEPS.compute_gs_energy(A, H, cfg)

A = load("simulation/ising_test_D2_X64/gs.jld2", "A");

using NPZ
A = npzread("one_h3_X96.npz")["A"] 
A = permutedims(A,(5,4,3,2,1))
A = permutedims(A,(3,4,1,2,5))

jldsave("gs.jld2", A = A)


