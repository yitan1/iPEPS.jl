using CairoMakie
s1 = iPEPS.Sz
s2 = iPEPS.SI
op1 = iPEPS.tout(s1, s2)
op2 = iPEPS.tout(s2, s1)
pxs, pys = make_es_path()
n = 20
px, py = pxs[n], pys[n]
envB1, basis = compute_spec_env(op1, px, py, "");
envB2, _ = compute_spec_env(op2, px, py, "");

es, vecs, P = compute_es(px, py, "");
exci_n = basis*P*vecs;
wka = exci_n' * envB1[:];
wkb = exci_n' * envB2[:];
a = exp(-im*(px+py)/3)
swk0 = wka.* conj(wka) + wkb.*conj(wkb) + wka.*conj(wkb).*a + wkb .*conj(wka).*conj(a) .|> real; # exp 

scatter(es[2:end-1], swk0[2:end-1])

x, y = plot_spectral(es[1:end], swk0[1:end], factor = 0.1)
lines(x, y)