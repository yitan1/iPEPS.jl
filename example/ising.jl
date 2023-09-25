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
