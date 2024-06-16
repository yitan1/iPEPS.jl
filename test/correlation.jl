using iPEPS
using TOML
using Test
using iPEPS: sigmaz, tout

cfg = TOML.parsefile("src/optimize/default_config.toml");
cfg["model"] = "hb"
op = tout(iPEPS.sigmaz, iPEPS.sI)
op1 = tout(iPEPS.sigmax, iPEPS.sigmax)
s2s, m1, m2 = iPEPS.correlation_spin(op1, cfg, max_iter = 20, direction = "v")

A = init_hb_gs(4)
s2s, m1, m2 = iPEPS.correlation_spin_A(A, op, cfg, max_iter = 20)