using iPEPS
using ConstructionBase

H = iPEPS.honeycomb(0.15, 0.15);

A = iPEPS.init_gs() .|> real ;

D = 2
d = 4
A = randn(D,D,D,D,d);

B = randn(size(A));

ts0 = iPEPS.CTMTensors(A,A);
ts1 = setproperties(ts0, B = B, Bd = conj(B));
# conv_fun(_x) = iPEPS.get_gs_energy(H,_x)
ts2, s = iPEPS.run_ctm(ts1, 50);
iPEPS.get_es_energy(H, ts2)
iPEPS.get_gs_energy(H, ts2)

# conv_fun(_x) = iPEPS.get_gs_energy(H,_x)
# ts0, s = iPEPS.run_ctm(ts0, 30, conv_fun = conv_fun);
# iPEPS.run_energy(H, ts0, 30, A)