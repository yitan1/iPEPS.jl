using iPEPS
using Zygote
using ConstructionBase
using TOML
using Test

### Test autodiff and no autodiff

H = ising();
cfg = TOML.parsefile("src/optimize/default_config.toml");

prepare_basis(H, cfg);

px, py = make_es_path()
for i in eachindex(px)
    optim_es(px[i], py[i], cfg)
end

E = plot_band(1, cfg)