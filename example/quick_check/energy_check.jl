using JLD2, TOML
A = load("simulation/hb_g11_D2_X64/gs.jld2", "A")
A = A ./ maximum(abs,A);

cfg = TOML.parsefile("src/default_config.toml");
iPEPS.compute_gs_energy(A, H, cfg)

A1 = iPEPS.get_symmetry(A)
A1[findall(x -> x< 1e-16, A)] .= 0

iPEPS.compute_gs_energy(A1, H, cfg)
# jldsave("gs.jld2", A = A)

t = iPEPS.CTMTensors(A, cfg)
conv_fun(_x) = iPEPS.get_gs_energy(_x, H)[1]
ts, _ = iPEPS.run_ctm(t; conv_fun = conv_fun);
iPEPS.get_gs_energy(ts, H)