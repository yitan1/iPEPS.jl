"""
    Tensor Order


- big -> small 
- bra -> ket
- top -> bottom
- left -> right
- out -> in
"""
module iPEPS

# using MKL # shoule be the first pkg
# using Accessors
using ConstructionBase
using TOML
using JLD2
using Printf
using LinearAlgebra
using OMEinsum
using Zygote, ChainRulesCore
using Optim#, LineSearches

# __precompile__(false)

export ising, heisenberg, honeycomb, hb_xx_k
export init_hb_gs
export optim_gs, prepare_basis, optim_es, optim_wp, make_es_path, plot_band, plot_spectral, basis_dep
export evaluate_wp, run_wp, compute_gs_energy, compute_es, compute_spec_env

# utility
include("utility/wrapper.jl")
include("utility/io.jl")
include("utility/autodiff.jl")
include("utility/config.jl")

# nested_tensor
include("nested_tensor/nested_tensor.jl")
include("nested_tensor/emptyT.jl")

# model
include("model/model.jl")
include("model/basis.jl")
include("model/ising.jl")
include("model/heisenberg.jl")
include("model/honeycomb.jl")
include("model/hb_B.jl")
# include("model/pepo.jl")

# ctm
include("ctm/ctm_tensor.jl")
include("ctm/ctmrg.jl")
# include("ctm/@ctmrgstep.jl")

# graph
include("graph/expectation.jl")
include("graph/evaluation.jl")

# optimize
include("optimize/optim_gs.jl")
include("optimize/optim_es.jl")
include("optimize/optim_wp.jl")


include("basis.jl")

include("plot.jl")
include("_test.jl")

# function __init__()
#     if ispath("config.toml")
#         cur_path = pwd()
#         println("Exist custom config file in $cur_path")
#     else
#         default_path = "$(@__DIR__)/default_config.toml"
#         println("Custom config is NOT EXIST! It will load daufult config file in $default_path")
#     end

#     nothing
# end

end
