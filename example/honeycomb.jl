using iPEPS
using MKL
using Random  

rng = MersenneTwister(10);
D = 2
A = randn(rng, Float64, D,D,D,D,4);

A = iPEPS.init_hb_gs(2) .|> real
H = iPEPS.honeycomb(1, 1);
res = optim_gs(H, A, "")

prepare_basis(H, "")
optim_es(0., 0., "")

using CairoMakie
s1 = iPEPS.Sz
s2 = iPEPS.SI
op = iPEPS.tout(s1, s2)
es, swk0 = compute_spectral(op, .0, .0, "")
scatter(es,swk0)

x, y = plot_spectral(es, swk0, factor = 0.1)
lines(x, y)

using JLD2, TOML
A = load("simulation/hb_test_D2_X64/gs.jld2", "A");
A = A ./ maximum(A);
cfg = TOML.parsefile("src/default_config.toml");
iPEPS.compute_gs_energy(A, H, cfg)

using NPZ
A = npzread("one_h3_X96.npz")["A"] 
A = permutedims(A,(5,4,3,2,1))
A = permutedims(A,(3,4,1,2,5))
cfg = TOML.parsefile("src/default_config.toml");
iPEPS.compute_gs_energy(A, H, cfg)
jldsave("gs.jld2", A = A)


