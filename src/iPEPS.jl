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

export optim_gs, prepare_basis, optim_es, init_hb_gs, make_es_path, plot_band, plot_spectral, compute_spectral, basis_dep

export ising, heisenberg, honeycomb

include("printing.jl")
include("io.jl")
include("tcon.jl")
include("model.jl")
include("svd_ad.jl")
include("basis.jl")
include("emptyT.jl")
include("nested_tensor.jl")
include("ctm_tensor.jl")
include("ctmrg.jl")
include("evaluation.jl")
# include("getCT.jl")
include("optim_gs.jl")
include("optim_es.jl")
include("expectation.jl")
include("plot.jl")
include("test.jl")

include("pepo.jl")

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
