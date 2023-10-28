using Test
using iPEPS
using iPEPS: honeycomb, init_hb_gs
using iPEPS: check_Q_op, get_ghz_111, get_ghz, get_Q_op, get_Q_ghz
using iPEPS: sigmax, sigmay, sigmaz
using OMEinsum
using TOML
using ConstructionBase

H = honeycomb(1, 1)
H = load("simulation/hb_g11_D2_X32/basis.jld2")["H"]
A = init_hb_gs(4)
cfg = TOML.parsefile("src/default_config.toml")
cfg["max_iter"] = 60
ts = iPEPS.CTMTensors(A, cfg);

# @ein B[m1, m2, m3, m4, m5] := A[p1, m2, m3, m4, m5] * iPEPS.sigmaz[p1, m1]
B = iPEPS.get_vison(2)
B = ts.A
ts1 = setproperties(ts, B=B, Bd=conj(B));

conv_fun(_x) = iPEPS.get_es_energy(_x, H) / iPEPS.get_all_norm(_x)[1]
ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);

bn, envB = iPEPS.get_all_norm(ts1); 
B2 = iPEPS.get_vison(2)
Complex.(B2[:])'*envB[:]

iPEPS.get_gs_energy(ts1, H)
iPEPS.get_gs_norm(ts1)