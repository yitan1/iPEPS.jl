"""
    Tensor Order


- big -> small 
- bra -> ket
- top -> bottom
- left -> right
- out -> in
"""
module iPEPS

# using Accessors
using ConstructionBase
using LinearAlgebra
using OMEinsum 
using Zygote
using Optim


# __precompile__(false)

# export CTM, updateCTM
# include("sym_ctmrg_gs/ctmrg.jl")

# export op_expect
# include("sym_ctmrg_gs/obs.jl")

# export sym_optimize_GS
# include("sym_ctmrg_gs/optim_GS.jl")

# # Excited States
# include("old_ctmrg/peps.jl")

# include("old_ctmrg/env_tensor.jl")

# include("old_ctmrg/env_tensor_exc.jl")

# #GS
# include("old_ctmrg/autodiff.jl")
# include("old_ctmrg/optim_GS.jl")
# #GS

# include("old_ctmrg/optim_ES.jl")

# include("old_ctmrg/op_expect.jl")
# include("old_ctmrg/op_expect2.jl")

# include("old_ctmrg/contraction.jl")

include("basis.jl")
include("ctm_tensor.jl")
include("emptyT.jl")
include("nested_tensor.jl")
include("tcon.jl")
include("ctmrg.jl")
# include("getCT.jl")
include("optim_gs.jl")

end
