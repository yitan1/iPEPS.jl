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

E = plot_band(4, "")
iPEPS.evaluate_es(0,0,"")

using CairoMakie
x1 = 1:10
x2 = 11.4:1.4:(11.4+9*1.4)
x3 = 25:34
x4 = 35.4:1.4:(35.4+3*1.4)

x = vcat(x1, x2, x3, x4)

f = Figure()
ax = Axis(f[1, 1])
for i = 1:3
    # x = collect(1:size(E,2))
    y = E[i,:]
    lines!(ax, x, y)
end
f


using NPZ, JLD2
bs = npzread("test_basis.npz")["basis"];
v1 = reshape(bs[:, 9], 2,2,2,2,2)
v1 = permutedims(v1, (3,4,5,2,1));

A =  load("simulation/ising_default_D2_X50/gs.jld2", "A");
# A = iPEPS.renormalize(A);
ts0 = iPEPS.CTMTensors(A, cfg);
conv_fun(_x) = iPEPS.get_gs_energy(_x, H)[1];
ts, _ = iPEPS.run_ctm(ts, conv_fun = conv_fun);

B = reshape(bs1[:,1], size(A));
ts1 = setproperties(ts0, B = B, Bd = conj(B));
ts1.Params["px"] = 0.3*pi;
ts1.Params["px"] = 0.3*pi;
@time ts11, _ = iPEPS.run_ctm(ts1);

n0 ,nb = iPEPS.get_all_norm(ts11);
n01, nb1 = iPEPS.get_all_norm1(ts11);

_, dA = iPEPS.get_tangent_basis(ts);
dA[:]'*v1[:]

bs1 = load("simulation/ising_default_D2_X50/basis.jld2", "basis");