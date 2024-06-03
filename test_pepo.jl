####
H0 = iPEPS.ising1(2.5)
I0 = iPEPS.get_identity()
pepoI = iPEPS.init_pepo(I0, 0.001);
pepoN = iPEPS.init_pepo(H0, 0.0001);

using JLD2, TOML
A = load("simulation/ising_default_D2_X30/gs.jld2", "A")
A = A ./ maximum(A)
cfg = TOML.parsefile("src/default_config.toml");

ts, ots = iPEPS.finite_ctm(H0, A, cfg);
ots1, ts1, _ = iPEPS.run_ctm(ots, ts);


ts = iPEPS.CTMTensors(A, cfg);
its = iPEPS.get_ots(ts, pepoI);
ots = iPEPS.get_ots(ts,pepoN);

tst = iPEPS.run_ctm(ts);
otst = iPEPS.run_ctm(ots);
ts1, ots1, _ = iPEPS.run_ctm(ts, ots);

iPEPS.compute_gs_energy(A, H, cfg)

e = iPEPS.get_gs_norm(ots1) /(1+2*20)^2
nrm = iPEPS.get_gs_norm(ts1)
A = A./nrm

log(e)/0.001 / iPEPS.get_gs_norm(ts1)   