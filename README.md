# iPEPS

## Introduction of some file:
- iPEPS.jl: main file

- Struct file: 

    1. ipeps.jl   
    2. ctm_tensor.jl: Struct of Corner transfer matrix
    3. nested_tensor.jl: tensor with [T, T_B, T_Bd, T_Bd]

- Basis file: 

    1. model.jl: construct Hamiltonian

- Optimization file:

    1. optim.jl: optimize the ground state and excite state
    2. ctmrg.jl: update ctm_tensor
    3. contraction.jl: evaluate some tensor contraction
