using iPEPS
using Zygote
using ConstructionBase
using TOML

H = iPEPS.honeycomb(0.15, 0.15);

A = iPEPS.init_hb_gs() .|> real;

D = 2
d = 4
chi = 30
A = randn(D,D,D,D,d);

B = randn(size(A));

H, N = iPEPS.optim_es(ts1, H, pi/5, pi/5)

cfg = TOML.parsefile("src/default_config.toml");

ts0 = iPEPS.CTMTensors(A,cfg);
ts0 = iPEPS.normalize_gs(ts0);
H = iPEPS.substract_gs_energy(ts0, H);
ts1, s = iPEPS.run_ctm(ts0);

basis = iPEPS.get_tangent_basis(ts0);
B = reshape(basis[:,6], size(A));

basis' * B[:]

ts1 = setproperties(ts0, B = B, Bd = conj(B));
using ProfileView 
iPEPS.get_es_grad(H, ts0.A, basis[:,1], 1.0,1.0)

ts2, s = iPEPS.run_ctm(ts1);
@time (y, ts), back = pullback(x -> iPEPS.run_es(ts2, H, x), B);
@time back((1,nothing));

iPEPS.get_es_energy(ts2, H)
iPEPS.get_gs_energy(ts2, H)

# conv_fun(_x) = iPEPS.get_gs_energy(H,_x)
# ts0, s = iPEPS.run_ctm(ts0, 30, conv_fun = conv_fun);
# iPEPS.run_energy(H, ts0, 30, A)

