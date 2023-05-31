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

optim_es(H, 0.5,0.5, "")
iPEPS.evaluate_es(0.5,0.5,"")


using NPZ, JLD2

effH = npzread("31_0.5_0.5.npz")["H"]
effN = npzread("31_0.5_0.5.npz")["N"]

effH = load("simulation/ising_default_D2_X30/es_0.5_0.5.jld2", "effH")
effN = load("simulation/ising_default_D2_X30/es_0.5_0.5.jld2", "effN")

H = (effH + effH') /2 
N = (effN + effN') /2
ev_N, P = eigen(N)

idx = sortperm(real.(ev_N))[end:-1:1]
ev_N = ev_N[idx]
selected = (ev_N/maximum(ev_N) ) .> 1e-3
P = P[:,idx]
P = P[:,selected]
N2 = P' * N * P
H2 = P' * H * P
H2 = (H2 + H2') /2 
N2 = (N2 + N2') /2
es, vecs = eigen(H2,N2)
ixs = sortperm(real.(es))
es = es[ixs]
vecs = vecs[:,ixs]

es, vecs