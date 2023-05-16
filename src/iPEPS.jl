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

end
