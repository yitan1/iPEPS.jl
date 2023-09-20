using CairoMakie
s1 = iPEPS.Sy
s2 = iPEPS.SI
op1 = iPEPS.tout(s1, s2)
op2 = iPEPS.tout(s2, s1)
pxs, pys = make_es_path()
n = 19
px, py = pxs[n], pys[n]
envB1, basis = compute_spec_env(op1, px, py, "");
envB2, _ = compute_spec_env(op2, px, py, "");

es, vecs, P = compute_es(px, py, "");
exci_n = basis*P*vecs;
wka = exci_n' * envB1[:];
wkb = exci_n' * envB2[:];
a = exp(-im*(px+py)/3)
swk0 = wka.* conj(wka) + wkb.*conj(wkb) + wka.*conj(wkb).*a + wkb .*conj(wka).*conj(a) .|> real; # exp 

x, y = plot_spectral(es[1:end], swk0[1:end], factor = 0.1)

f = Figure(xlabelfont = 34, ylabelfont = 34)
ax = Axis(f[1, 1], title = "plot", xlabel = L"{\omega}", ylabel = L"S^{yy}(0, \omega)")
lines!(ax, x, y, label = "Syy")
scatter!(ax, es[1:end], swk0[1:end], label = "Syy")
# axislegend()
xlims!(ax, low = -0.1, high = 8)
f

pxs, pys = make_es_path()
n = 19
px, py = pxs[n], pys[n]
step = 1
max_b = 55
f = Figure()
ax = Axis(f[1, 1])
for i = 1:step:max_b
    es, _ = iPEPS.basis_dep(i, px, py, "")
    scatter!(ax, ones(length(es))*i, es)
end
# ylims!(ax, low = -0.5, high = 6)
# xlims!(ax, low = 0, high = max_b)
f  