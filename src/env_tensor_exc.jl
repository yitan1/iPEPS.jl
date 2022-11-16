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

function get_envtensor(phi1::ExcIPEPS, phi2::ExcIPEPS; kwargs...)
    chi = get(kwargs, :chi, 100) # TODO: error when chi is not assigned
    env = init_env(phi1, phi2, chi)

    maxitr = get(kwargs, :maxitr, 1000)
    conv_tol = get(kwargs, :conv_tol, 1e-8)
    olds = zeros(eltype(env), chi)
    diff = 1.0

    output = get(kwargs, :output, true)
end

function init_env(phi1::ExcIPEPS, phi2::ExcIPEPS, chi)
    phi0 = get_GS(phi)

    env0 = init_env(phi0, chi)
    # TODO
end

function update_env(envs::ExcEnvTensor)
end

function up_left(envs::ExcEnvTensor)
    
end