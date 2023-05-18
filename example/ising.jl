
using iPEPS
# using MKL

H = ising()

A = randn(2,2,2,2,2);

res = optim_gs(H, A, "")

px, py = make_es_path()

for i in eachindex(px)
    optim_es(H, px[i], py[i], "")
end   

E = plot_band(6, "")

using CairoMakie

f = Figure()
ax = Axis(f[1, 1])
for i = 1:6
x = collect(1:size(E,2))
y = E[i,:]
lines!(ax, x, y)
end
f
