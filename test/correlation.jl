using iPEPS
using TOML
using Test
using iPEPS: sigmaz, tout

cfg = TOML.parsefile("src/optimize/default_config.toml");
cfg["model"] = "hb"
op = tout(iPEPS.sI, iPEPS.sigmax)
op1 = tout(iPEPS.sigmax, iPEPS.sI)
iPEPS.correlation_spin(op, cfg, max_iter = 20)