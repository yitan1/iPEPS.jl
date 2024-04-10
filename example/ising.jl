using iPEPS
using Random  
# using MKL
using TOML

function compute(h)
    cfg = TOML.parsefile("src/optimize/default_config.toml")
    cfg["out_prefix"] = "h$h"
    H = ising(h);
    rng = MersenneTwister(3);
    A = randn(rng, Float64, 2,2,2,2,2);
    res = optim_gs(H, A, cfg)

    if !res.ls_success
        println("Line search failed")
        return false
    end
    prepare_basis(H, cfg)
    optim_es(0, 0, cfg)
    return true
end

h = 2.8
cfg = TOML.parsefile("src/optimize/default_config.toml")
cfg["out_prefix"] = "h$h"
H = ising(h);
rng = MersenneTwister(2);
A = randn(rng, Float64, 2,2,2,2,2);
res = optim_gs(H, A, cfg; f_tol = 1e-6)

prepare_basis(H, cfg)
optim_es(0, 0, cfg)


cfg["out_prefix"] = "h0.1"
cfg["nrmB_cut"] = 6
es, _ = iPEPS.evaluate_es(0,0,cfg)

px, py = make_es_path()
for i in 1:34 
    # if i in [30]
    #     continue
    # end
    optim_es(px[i], py[i], "")
end
es, _ = iPEPS.evaluate_es(px[30], py[30], "")
optim_es(1,0,"")
es, _ = iPEPS.evaluate_es(0,0,"")



ts = load("simulation/ising_25_D2_X32/basis.jld2")["ts"];
basis = load("simulation/ising_25_D2_X32/basis.jld2", "basis")
H = load("simulation/ising_25_D2_X32/basis.jld2", "H")
ts.Params["max_iter"] = 60
ts.Params["chi"] = 32

es, vecs, P = compute_es(0, 0, ""; disp = true);
exci_n = basis*P*vecs;
# 
B1 = reshape(exci_n[:,2], size(ts.A))
# B1 = reshape(basis[:,20], size(ts.A))
# B1 = randn(size(ts.A))
ts1 = setproperties(ts, B=B1, Bd=conj(B1));
# conv_fun(_x) =  iPEPS.get_es_energy(_x, H) /iPEPS.get_all_norm(_x)[1]
conv_fun(_x) =  iPEPS.get_es_energy(_x, H) 
# conv_fun(_x) =  iPEPS.get_all_norm(_x)[1]
ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);
