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
using LinearAlgebra
using OMEinsum 
using Zygote
using Optim#, LineSearches

# __precompile__(false)

export optim_gs, optim_es, honeycomb, init_hb_gs

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

function __init__()
    if ispath("config.toml")
        cur_path = pwd()
        println("Exist custom config file in $cur_path")
    else
        default_path = "$(@__DIR__)/default_config.toml"
        println("Custom config is NOT EXIST! It will load daufult config file in $default_path")
    end

    nothing
end

end
