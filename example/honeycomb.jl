using iPEPS
# using MKL
using Random  

rng = MersenneTwister(10);
D = 2
A = randn(rng, Float64, D,D,D,D,4);

A = iPEPS.init_hb_gs(2, p1 = 0.24, p2 = 0) |> real;
H = iPEPS.honeycomb(0.15, 0.15);
res = optim_gs(H, A, "")

prepare_basis(H, "")
optim_es(0., 0., "");

using CairoMakie
s1 = iPEPS.Sy
s2 = iPEPS.SI
op1 = iPEPS.tout(s1, s2)
op2 = iPEPS.tout(s2, s1)
pxs, pys = make_es_path()
n = 20
px, py = pxs[n], pys[n]
envB1, basis = compute_spec_env(op1, px, py, "");
envB2, _ = compute_spec_env(op2, px, py, "");

es, vecs, P = compute_es(px, py, "");
exci_n = basis*P*vecs;
wka = exci_n' * envB1[:];
wkb = exci_n' * envB2[:];
a = exp(-im*(px+py)/3)
swk0 = wka.* conj(wka) + wkb.*conj(wkb) + wka.*conj(wkb).*a + wkb .*conj(wka).*conj(a) .|> real; # exp 

scatter(es[1:end], swk0[1:end])

x, y = plot_spectral(es[1:end], swk0[1:end], factor = 0.1)
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


H1 = load("simulation/hb_test_D2_X64/es_0_0.jld2", "effH");
H2 = load("simulation/hb_test_D2_X64/ades_0_0.jld2", "effH");
H2[1,:]/2 - H1[1,:]

t = iPEPS.CTMTensors(A, cfg)
conv_fun(_x) = iPEPS.get_gs_energy(_x, H)[1]
ts, _ = iPEPS.run_ctm(t; conv_fun = conv_fun)
iPEPS.get_gs_energy(ts, H)