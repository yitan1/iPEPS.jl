using iPEPS
using Random  
# using MKL

H = ising(2.5);

rng = MersenneTwister(3);

A = randn(rng, Float64, 2,2,2,2,2);

res = optim_gs(H, A, "")

px, py = make_es_path()

for i in eachindex(px)
    optim_es(H, px[i], py[i], "")
end   

E = plot_band(5, "")
iPEPS.evaluate_es(0,0,"")

using CairoMakie
x1 = 1:10
x2 = 11.4:1.4:(11.4+9*1.4)
x3 = 25:34
x4 = 35.4:1.4:(35.4+3*1.4)

x = vcat(x1, x2, x3, x4)

f = Figure()
ax = Axis(f[1, 1])
for i = 1:5
    # x = collect(1:size(E,2))
    y = E[i,:]
    lines!(ax, x, y)
end
f

op = iPEPS.Sz
es, swk0 = compute_spectral(op, .0, .0, "")

x, y = plot_spectral(es, swk0)

lines(x, y)
prepare_basis(H, "")
iPEPS.test_es(90, 0.1, 0.1, "")

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