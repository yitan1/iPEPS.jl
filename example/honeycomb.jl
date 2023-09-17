using iPEPS
# using MKL
using Random  

rng = MersenneTwister(4);
D = 2
A = randn(rng, Float64, D,D,D,D,4);
A = iPEPS.get_symmetry(A)

A = iPEPS.init_hb_gs(2, p1 = 0.24, p2 = 0);
jldsave("simulation/hb_g11_D2_X64/gs.jld2", A = A)
H = iPEPS.honeycomb(1, 1);
res = optim_gs(H .*2, A, "")

prepare_basis(H, "")
optim_es(0., 0., "");

using CairoMakie
s1 = iPEPS.Sz
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

scatter(es[2:end-1], swk0[2:end-1])

x, y = plot_spectral(es[1:end], swk0[1:end], factor = 0.1)
lines(x, y)

using JLD2, TOML
A = load("simulation/hb_g11_D2_X64/gs.jld2", "A")
A = A ./ maximum(abs,A);
cfg = TOML.parsefile("src/default_config.toml");
A = test()
iPEPS.compute_gs_energy(ts.A, H, cfg)
A1[findall(x -> x< 1e-16, A)] .= 0
A1 = iPEPS.get_symmetry(A)
iPEPS.compute_gs_energy(A1, H, cfg)
jldsave("gs.jld2", A = A)

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
ts, _ = iPEPS.run_ctm(t; conv_fun = conv_fun);
iPEPS.get_gs_energy(ts, H)


B = test() #|> iPEPS.renormalize
 
ts = load("simulation/hb_g11_D2_X64/basis.jld2")["ts"];
H = load("simulation/hb_g11_D2_X64/basis.jld2")["H"]
Cs, Es = iPEPS.init_ctm(ts.A, ts.Ad);
ts1 = setproperties(ts, Cs = Cs, Es = Es, B = B, Bd = conj(B), Params = cfg);

conv_fun(_x) = iPEPS.get_es_energy(_x, H)
ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);

iPEPS.get_gs_energy(ts1, H)
iPEPS.get_es_energy(ts1, H)
Nb, envB = iPEPS.get_all_norm(ts1);

Complex.(ts.A[:])'*envB[:]