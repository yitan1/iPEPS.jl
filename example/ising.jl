using iPEPS
using Random  
# using MKL

H = ising(2.5);

rng = MersenneTwister(3);

A = randn(rng, Float64, 2,2,2,2,2);

res = optim_gs(H, A, "")

prepare_basis(H, "")

px, py = make_es_path()
for i in 1:34 
    if i in [30, 5, 10, 15, 20, 25]
        continue
    end
    optim_es(px[i], py[i], "")
end
es, _ = iPEPS.evaluate_es(px[25], py[25], "")

n = 5
E = plot_band(n, "")

using CairoMakie
f = Figure()
ax = Axis(f[1, 1])
for i = 1:n
    # x = collect(1:size(E,2))
    y = E[i,:]
    lines!(ax, y)
end
f

op = iPEPS.Sz
es, swk0 = compute_spectral(op, .0, .0, "")

x, y = plot_spectral(es, swk0)
lines(x, y)

optim_es(0, 0, "")

ts = load("simulation/ising_25_D2_X32/basis.jld2")["ts"];
basis = load("simulation/ising_25_D2_X32/basis.jld2", "basis")
H = load("simulation/ising_25_D2_X32/basis.jld2", "H")
ts.Params["max_iter"] = 60
ts.Params["chi"] = 32

es, vecs, P = compute_es(0, 0, ""; disp = true);
exci_n = basis*P*vecs;
# 
B1 = reshape(exci_n[:,2], size(ts.A))
# B1 = reshape(basis[:,20], size(ts.A))
# B1 = randn(size(ts.A))
ts1 = setproperties(ts, B=B1, Bd=conj(B1));
# conv_fun(_x) =  iPEPS.get_es_energy(_x, H) /iPEPS.get_all_norm(_x)[1]
conv_fun(_x) =  iPEPS.get_es_energy(_x, H) 
# conv_fun(_x) =  iPEPS.get_all_norm(_x)[1]
ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);
