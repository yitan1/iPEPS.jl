using iPEPS
using ConstructionBase
using TOML, JLD2
B = iPEPS.get_vison(2) #|> iPEPS.renormalize

ts = load("simulation/hb_g11_D2_X32/basis.jld2")["ts"];
H = load("simulation/hb_g11_D2_X32/basis.jld2")["H"]
cfg = TOML.parsefile("src/default_config.toml");
Basis = load("simulation/hb_g11_D2_X32/basis.jld2")["basis"]
# B = reshape(Basis[:,1], size(ts.A))
# B = ts.A
Cs, Es = iPEPS.init_ctm(ts.A, ts.Ad);
cfg["px"] = convert(eltype(ts.A), 0.0*pi)
cfg["py"] = convert(eltype(ts.A), 0.0*pi)
ts1 = setproperties(ts, Cs = Cs, Es = Es, B = B, Bd = conj(B), Params = cfg);

conv_fun(_x) = iPEPS.get_es_energy(_x, H) / iPEPS.get_all_norm(_x)[1]
ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);

iPEPS.get_gs_energy(ts1, H)
iPEPS.get_es_energy(ts1, H)
Nb, envB = iPEPS.get_all_norm(ts1);
iPEPS.get_gs_norm(ts1)

Complex.(ts.A[:])'*envB[:]