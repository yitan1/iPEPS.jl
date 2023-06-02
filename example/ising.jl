using iPEPS
using Random
# using MKL

H = ising(2);

rng = MersenneTwister(3);

A = randn(rng, Float32, 2,2,2,2,2);

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
get_basis(H, "")
es, swk0 = compute_spectral(op, .0, .0, "")

x, y = plot_spectral(es, swk0)

lines(x, y)