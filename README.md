# iPEPS

## Introduction of some file:
- iPEPS.jl: main file


- Basis file: 

    1. model.jl: construct Hamiltonian
    2. svd_ad.jl: svd for zygote extend
    3. tcon.jl: wrap the contraction function

- Struct file: 

    1. ipeps.jl 
    2. emptyT.jl: convenient struct for contraction 
    3. nested_tensor.jl: tensor with [T, T_B, T_Bd, T_Bd]
    4. ctm_tensor.jl: Struct of Corner transfer matrix

- Optimization file:

    1. ctmrg.jl: update ctm_tensor
    2. optim_gs.jl: optimize the ground state and excited state
    3. optim_es.jl
