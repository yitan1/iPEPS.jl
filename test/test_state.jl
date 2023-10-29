using Test
using iPEPS
using iPEPS: honeycomb, init_hb_gs
using iPEPS: check_Q_op, get_ghz_111, get_ghz, get_Q_op, get_Q_ghz
using iPEPS: sigmax, sigmay, sigmaz
using OMEinsum
using TOML
using ConstructionBase

# H = honeycomb(1, 1)
H = load("simulation/hb_g11_D2_X32/basis.jld2")["H"]
# ii = Matrix{eltype(H[1])}(I, size(H[1]))
A = init_hb_gs(2)
cfg = TOML.parsefile("src/default_config.toml")
cfg["max_iter"] = 30
# ts = iPEPS.CTMTensors(A, cfg);
ts = load("simulation/hb_g11_D4_X64/basis.jld2")["ts"];

# @ein B[m1, m2, m3, m4, m5] := A[p1, m2, m3, m4, m5] * iPEPS.sigmaz[p1, m1]
# B1 = iPEPS.get_vison(4) 
B1 = randn(2,2,2,2,4);
B1 = iPEPS.act_R_op(B1)
# basis = load("simulation/hb_g11_D2_X32/basis.jld2", "basis")
# B1d = reshape(basis[:,5], size(ts.A))
# B1 = iPEPS.tcon([ts.A, op1], [[-1,-2,-3,-4,1], [-5,1]])
ts1 = setproperties(ts, B=B1, Bd=conj(B1));

conv_fun(_x) = iPEPS.get_es_energy(_x, H) / iPEPS.get_all_norm(_x)[1]
ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);

bn, envB = iPEPS.get_all_norm(ts1); 
# B2 = iPEPS.get_vison(2)
s1 = iPEPS.sigmaz
s2 = iPEPS.sigmaz
op1 = iPEPS.tout(s2, s1)
@ein B0[m1,m2,m3,m4,m5] := A[m1,m2,m3,m4,p1] * op1[p1,m5]

B2 = randn(ComplexF64, 2,2,2,2,4);
B2 = iPEPS.act_R_op(B2, add = 2)
Complex.(B2[:])'*envB[:]

iPEPS.get_gs_energy(ts1, H)
iPEPS.get_gs_norm(ts1)

load("simulation/ising_25_D2_X32/es_0_0.jld2")["effN"]