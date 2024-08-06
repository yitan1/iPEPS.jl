using iPEPS
using TOML
using Test
using iPEPS: sigmaz, tout

cfg = TOML.parsefile("src/optimize/default_config.toml");
cfg["model"] = "hb"
op = tout(iPEPS.sigmaz, iPEPS.sI)
op1 = tout(iPEPS.sigmax, iPEPS.sigmax)
s2s, m1, m2 = iPEPS.correlation_spin(op1, cfg, max_iter = 20, direction = "h")

A = init_hb_gs(4, dir = "ZZ")
s2s, m1, m2 = iPEPS.correlation_spin_A(A, op, cfg, max_iter = 20)

A = randn(ComplexF64, 4,4,4,4,4)

cfg["max_iter"] = 100
cfg["chi"] = 20
es, es_o, TM = iPEPS.correlation_TM_A(A, cfg, direction = "h")
r = log.(es[1] ./ es) 

rl, im = real(es_o), imag(es_o)