"""
    ExcEnvTensor

1. bulk: A_Ad, B_Ad, A_Bd, B_Bd
2. corner: Ci, B_Ci, Bd_Ci, BB_Ci;  
3. edge: Ei, B_Ei, Bd_Ei, BB_Ei;
    (where i = 1, 2, 3, 4)
"""
struct ExcEnvTensor 
    data::Vector{EnvTensor}
end

function get_envtensor(phi::ExcIPEPS)
    
end

function init_env(phi::ExcIPEPS, chi)
    phi0 = get_GS(phi)

    env0 = init_env(get_GS(phi), chi)
    # TODO
end

function update_env!(envs::ExcEnvTensor)
end

function up_left!(envs::ExcEnvTensor)
    
end