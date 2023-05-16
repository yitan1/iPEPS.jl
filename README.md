# iPEPS

## Introduction of some file:
- iPEPS.jl: main file


- Basis file: 
          
    1. io.jl: create dir and file 
    2. tcon.jl: wrap the contraction function
    3. model.jl: construct Hamiltonian
    4. svd_ad.jl: svd for zygote extend
    5. basis.jl: 

- Struct file: 

    1. emptyT.jl: convenient struct for contraction 
    2. nested_tensor.jl: tensor with [T, T_B, T_Bd, T_Bd]
    3. ctm_tensor.jl: Struct of Corner transfer matrix

- Optimization file:

    1. ctmrg.jl: update ctm_tensor
    2. optim_gs.jl: optimize the ground state 
    3. optim_es.jl: optimize the excited state
