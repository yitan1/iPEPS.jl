"""
    ExcEnvTensor

1. bulk:   A_Ad,  B_Ad,  A_Bd,   B_Bd
2. corner: Ci,    B_Ci,  Bd_Ci,  BB_Ci;  
3. edge:   Ei,    B_Ei,  Bd_Ei,  BB_Ei;
    (where i = 1, 2, 3, 4)
"""
struct ExcEnvTensor 
    data::Vector{EnvTensor}
end

Base.getindex(envs::ExcEnvTensor, i::Int) = envs.data[i]

ExcEnvTensor(args...) = ExcEnvTensor(collect(args))

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
    # get_GS(phi1) == get_GS(phi2)
    env1 = init_env(get_GS(phi1), chi)  
    B_Ad = transfer_matrix(get_B(phi1), get_Ad(phi2))
    A_Bd = transfer_matrix(get_A(phi1), get_Bd(phi2))
    B_Bd = transfer_matrix(get_B(phi1), get_Bd(phi2))

    Cs = corner(env1)
    Es = edge(env1)
    env2 = EnvTensor(B_Ad, deepcopy(Cs), deepcopy(Es), chi) # XXX deepcopy -> copy ?
    env3 = EnvTensor(A_Bd, deepcopy(Cs), deepcopy(Es), chi)
    env4 = EnvTensor(B_Bd, deepcopy(Cs), deepcopy(Es), chi)
    exc_env = ExcEnvTensor(env1, env2, env3, env4)

    exc_env
end

function update_env(envs::ExcEnvTensor)
    envs, s = up_left(envs)
end

function up_left(envs::ExcEnvTensor)
    Pl, Pld, s = left_projector(envs[1]) 

    env1 = up_left(envs[1], Pl, Pld) # up: C1,    E4,    C4
    env2 = up_left_B(envs, Pl, Pld)  # up: B_C1,  B_E4,  B_C4
    env3 = up_left_Bd(envs, Pl, Pld) # up: Bd_C1, Bd_E4, Bd_C4
    env4 = up_left_BB(envs, Pl, Pld) # up: BB_C1, BB_E4, BB_C4

    envs = ExcEnvTensor(env1, env2, env3, env4)
    envs
end

function up_left_B(envs::EnvTensor, Pl, Pld)
    
end