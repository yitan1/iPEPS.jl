using CairoMakie
s1 = iPEPS.sigmaz
s2 = iPEPS.sI
op1 = iPEPS.tout(s1, s2)
op2 = iPEPS.tout(s2, s1)
pxs, pys = make_es_path()
n = 20
px, py = pxs[n], pys[n]

ts = load("simulation/hb_g11_D2_X32/basis.jld2")["ts"];
# basis = load("simulation/hb_g11_D2_X32/basis.jld2", "basis")
# H = load("simulation/hb_g11_D2_X32/basis.jld2", "H")
# ts.Params["max_iter"] = 30
# ts.Params["chi"] = 96

es, vecs, P, envB = compute_es(px, py, ""; disp = true);
# exci_n = basis*P*vecs;

# B1 = reshape(exci_n[:,20], size(ts.A))
# # B1 = reshape(basis[:,10], size(ts.A))
# # B1 = iPEPS.tcon([A, op1], [[-1,-2,-3,-4,1], [-5,1]])
# ts1 = setproperties(ts, B=B1, Bd=conj(B1));
# conv_fun(_x) = iPEPS.get_all_norm(_x)[1] 
# # conv_fun(_x) = get_es_energy(_x, H) / get_all_norm(_x)[1]
# ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);

# _, env_B =iPEPS.get_all_norm(ts1)

op1A = iPEPS.tcon([ts.A, op1], [[-1,-2,-3,-4,1], [-5,1]])
op2A = iPEPS.tcon([ts.A, op2], [[-1,-2,-3,-4,1], [-5,1]]);

wka = op1A[:]'*envB*P*vecs
wkb = op2A[:]'*envB*P*vecs

a = exp(-im*(px+py)/3)
swk0 = wka.* conj(wka) + wkb.*conj(wkb) + wka.*conj(wkb).*a + wkb .*conj(wka).*conj(a) .|> real; # exp 

x, y = plot_spectral(es[1:end], swk0[1:end], factor = 0.2)

f = Figure(xlabelfont = 34, ylabelfont = 34)
ax = Axis(f[1, 1], title = "plot", xlabel = L"{\omega}", ylabel = L"S^{yy}(0, \omega)")
lines!(ax, x, y, label = "Syy")
scatter!(ax, es[1:end], swk0[1:end], label = "Syy")
# axislegend()
# xlims!(ax, low = -1, high = 6)
f

