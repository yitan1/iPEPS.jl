using iPEPS
using JLD2, TOML
using Test

A = init_hb_gs(2, p1 = 0.24, p2 = 0, dir ="XX")
cfg = TOML.parsefile("src/optimize/default_config.toml")
cfg["model"] = "hb"
gs_name = iPEPS.get_gs_name(cfg)
jldsave(gs_name; A=A)

H = honeycomb(1, 1, dir = "XX")
prepare_basis(H, cfg)

## 
w_op = iPEPS.get_w_op()
cfg["basis_name"] = "basis"
optim_wp(w_op, cfg)
get_wp_basis(cfg)

## part save
cfg["basis_name"] = "wp/basis_1m"
cfg["wp"] = 2
cfg["es_resume"] = 1
cfg["es_num"] = 10
cfg["part_save"] = true
optim_wp(w_op, cfg)
get_wp_basis(cfg)