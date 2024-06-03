using iPEPS
using TOML
using Profile
using BenchmarkTools

A = init_hb_gs(2, p1=0.24, p2=0, dir="XX")
H = honeycomb(1, 1, dir="XX")
cfg = TOML.parsefile("src/optimize/default_config.toml")

@time ts = iPEPS.CTMTensors(A, cfg);



