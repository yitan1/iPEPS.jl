using iPEPS
using Random  
# using MKL

H = ising(2.5);

rng = MersenneTwister(3);

A = randn(rng, Float64, 2,2,2,2,2);

res = optim_gs(H, A, "")

prepare_basis(H, "")

px, py = make_es_path()
for i in eachindex(px)
    optim_es(px[i], py[i], "")
end   

E = plot_band(5, "")
iPEPS.evaluate_es(0,0,"")

using CairoMakie
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
