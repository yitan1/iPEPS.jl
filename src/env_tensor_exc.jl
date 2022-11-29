"""
    ExcEnvTensor

1. bulk:   A_Ad,  B_Ad,  A_Bd,   B_Bd
2. corner: Ci,    Ci_B,  Ci_Bd,  Ci_BB;  
3. edge:   Ei,    Ei_B,  Ei_Bd,  Ei_BB;
    (where i = 1, 2, 3, 4)
"""
struct ExcEnvTensor 
    data::Vector{EnvTensor}
end

Base.getindex(envs::ExcEnvTensor, i::Int) = envs.data[i]

ExcEnvTensor(args...) = ExcEnvTensor(collect(args))

function get_envtensor(phi1::ExcIPEPS, phi2::ExcIPEPS; kwargs...) 
    chi = get(kwargs, :chi, 10) 
    kx, ky = get_kx(phi1), get_ky(phi1)

    # init
    envs = init_env(phi1, phi2, chi)
    olds = zeros(chi)
    diff = 1.0

    # parameter 
    maxitr = get(kwargs, :maxitr, 1000)
    conv_tol = get(kwargs, :conv_tol, 1e-8)
    output = get(kwargs, :output, true)

    for i = 1:maxitr
        envs, s = update_env(envs, kx, ky)

        if length(s) == length(olds)
            diff = norm(s - olds)
            if output == true
                @show i, diff
            end
        end
        if diff < conv_tol
            break
        end
        olds = s
    end

    envs
end

function init_env(phi1::ExcIPEPS, phi2::ExcIPEPS, chi)
    # get_GS(phi1) == get_GS(phi2)
    env1 = init_env(get_GS(phi1), chi)  
    B_Ad = transfer_matrix(get_B(phi1), get_Ad(phi2))
    A_Bd = transfer_matrix(get_A(phi1), get_Bd(phi2))
    B_Bd = transfer_matrix(get_B(phi1), get_Bd(phi2))

    Cs = corner(env1)
    Es = edge(env1)
    env2 = EnvTensor(B_Ad, deepcopy(Cs), deepcopy(Es), chi) 
    env3 = EnvTensor(A_Bd, deepcopy(Cs), deepcopy(Es), chi)
    env4 = EnvTensor(B_Bd, deepcopy(Cs), deepcopy(Es), chi)
    exc_env = ExcEnvTensor(env1, env2, env3, env4)

    exc_env
end

function update_env(envs::ExcEnvTensor, kx, ky)
    envs, s1 = up_left(envs, kx)
    envs, __ = up_right(envs, kx)
    envs, __ = up_top(envs, ky)
    envs, __ = up_bottom(envs, ky)

    envs, s1
end

function up_left(envs::ExcEnvTensor, kx)
    Pl, Pld, s = left_projector(envs[1]) 

    env1 = up_left(envs[1], Pl, Pld) 
    env2 = up_left_B(envs, kx, Pl, Pld)  
    env3 = up_left_Bd(envs, kx, Pl, Pld) 
    env4 = up_left_BB(envs, Pl, Pld) 

    envs = ExcEnvTensor(env1, env2, env3, env4)
    envs, s
end

function up_right(envs::ExcEnvTensor, kx)
    Pr, Prd, s = right_projector(envs[1]) 

    env1 = up_right(envs[1], Pr, Prd)     
    env2 = up_right_B(envs, kx, Pr, Prd)  
    env3 = up_right_Bd(envs, kx, Pr, Prd) 
    env4 = up_right_BB(envs, Pr, Prd)     

    envs = ExcEnvTensor(env1, env2, env3, env4)
    envs, s
end

function up_top(envs::ExcEnvTensor, ky)
    Pt, Ptd, s = top_projector(envs[1]) 

    env1 = up_top(envs[1], Pt, Ptd)     
    env2 = up_top_B(envs, ky, Pt, Ptd)  
    env3 = up_top_Bd(envs, ky, Pt, Ptd) 
    env4 = up_top_BB(envs, Pt, Ptd)     

    envs = ExcEnvTensor(env1, env2, env3, env4)
    envs, s
end

function up_bottom(envs::ExcEnvTensor, ky)
    Pb, Pbd, s = bottom_projector(envs[1]) 

    env1 = up_bottom(envs[1], Pb, Pbd)     
    env2 = up_bottom_B(envs, ky, Pb, Pbd)  
    env3 = up_bottom_Bd(envs, ky, Pb, Pbd) 
    env4 = up_bottom_BB(envs, Pb, Pbd)     

    envs = ExcEnvTensor(env1, env2, env3, env4)
    envs, s
end

"""
    up_left_B

Return C1_B, E4_B, C4_B
"""
function up_left_B(envs::ExcEnvTensor, kx, Pl, Pld)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_B, Es_B, T_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])

    newC1_B, newE4_B, newC4_B = proj_left_B(kx, Pl, Pld, Cs_B[1], Es[1], Cs[1], Es_B[1], 
                                                         Es_B[4], T, Es[4], T_B, 
                                                         Cs_B[4], Es[3], Cs[4], Es_B[3]) # XXX: unnecessay computation

    env2 = EnvTensor(T_B, [newC1_B, Cs_B[2], Cs_B[3], newC4_B], [Es_B[1], Es_B[2], Es_B[3], newE4_B], get_maxchi(envs[2]))
    env2
end

"""
    up_left_Bd

Return C1_Bd, E4_Bd, C4_Bd
"""
function up_left_Bd(envs::ExcEnvTensor, kx, Pl, Pld)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_Bd, Es_Bd, T_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])

    newC1_Bd, newE4_Bd, newC4_Bd = proj_left_Bd(kx, Pl, Pld, Cs_Bd[1], Es[1], Cs[1], Es_Bd[1], 
                                                             Es_Bd[4], T, Es[4], T_Bd, 
                                                             Cs_Bd[4], Es[3], Cs[4], Es_Bd[3]) 
    
    env3 = EnvTensor(T_Bd, [newC1_Bd, Cs_Bd[2], Cs_Bd[3], newC4_Bd], [Es_Bd[1], Es_Bd[2], Es_Bd[3], newE4_Bd], get_maxchi(envs[3]))
    env3
end

"""
    up_left_BB

Return C1_BB, E4_BB, C4_BB
"""
function up_left_BB(envs::ExcEnvTensor, Pl, Pld)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_B, Es_B, T_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])
    Cs_Bd, Es_Bd, T_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])
    Cs_BB, Es_BB, T_BB = corner(envs[4]), edge(envs[4]), bulk(envs[4])

    newC1_BB, newE4_BB, newC4_BB = proj_left_BB(Pl, Pld, 
                                    Cs_BB[1],Es[1], Cs[1],Es_BB[1], Cs_B[1],Es_Bd[1], Cs_Bd[1], Es_B[1],
                                    Es_BB[4],T,     Es[4],T_BB,     Es_B[4],T_Bd,     Es_Bd[4], T_B,
                                    Cs_BB[4],Es[3], Cs[4],Es_BB[3], Cs_B[4],Es_Bd[3], Cs_Bd[4], Es_B[3]) 
    
    env4 = EnvTensor(T_BB, [newC1_BB, Cs_BB[2], Cs_BB[3], newC4_BB], [Es_BB[1], Es_BB[2], Es_BB[3], newE4_BB], get_maxchi(envs[4]))
    env4
end

"""
    up_right_B

Return C2_B, E2_B, C3_B
"""
function up_right_B(envs::ExcEnvTensor, kx, Pr, Prd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_B, Es_B, T_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])

    newC2_B, newE2_B, newC3_B = proj_right_B(kx, Pr, Prd, Cs_B[2], Es[1], Cs[2], Es_B[1], 
                                                          Es_B[2], T,     Es[2], T_B, 
                                                          Cs_B[3], Es[3], Cs[3], Es_B[3]) # XXX: unnecessay computation

    env2 = EnvTensor(T_B, [Cs_B[1], newC2_B, newC3_B, Cs_B[4]], [Es_B[1], newE2_B, Es_B[3], Es_B[4]], get_maxchi(envs[2]))
    env2
end

"""
    up_right_Bd

Return C2_Bd, E2_Bd, C3_Bd
"""
function up_right_Bd(envs::ExcEnvTensor, kx, Pr, Prd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_Bd, Es_Bd, T_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])

    newC2_Bd, newE2_Bd, newC3_Bd = proj_right_Bd(kx, Pr, Prd, Cs_Bd[2], Es[1], Cs[2], Es_Bd[1], 
                                                              Es_Bd[2], T,     Es[2], T_Bd, 
                                                              Cs_Bd[3], Es[3], Cs[3], Es_Bd[3]) 
    
    env3 = EnvTensor(T_Bd, [Cs_Bd[1], newC2_Bd, newC3_Bd, Cs_Bd[4]], [Es_Bd[1], newE2_Bd, Es_Bd[3], Es_Bd[4]], get_maxchi(envs[3]))
    env3
end

"""
    up_right_BB

Return C2_BB, E2_BB, C3_BB
"""
function up_right_BB(envs::ExcEnvTensor, Pr, Prd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_B, Es_B, T_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])
    Cs_Bd, Es_Bd, T_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])
    Cs_BB, Es_BB, T_BB = corner(envs[4]), edge(envs[4]), bulk(envs[4])

    newC2_BB, newE2_BB, newC3_BB = proj_right_BB(Pr, Prd, 
                                    Cs_BB[2],Es[1], Cs[2],Es_BB[1], Cs_B[2],Es_Bd[1], Cs_Bd[2], Es_B[1],
                                    Es_BB[2],T,     Es[2],T_BB,     Es_B[2],T_Bd,     Es_Bd[2], T_B,
                                    Cs_BB[3],Es[3], Cs[3],Es_BB[3], Cs_B[3],Es_Bd[3], Cs_Bd[3], Es_B[3]) 

    env4 = EnvTensor(T_BB, [Cs_BB[1], newC2_BB, newC3_BB, Cs_BB[4]], [Es_BB[1], newE2_BB, Es_BB[3], Es_BB[4]], get_maxchi(envs[4]))
    env4
end

"""
    up_top_B

Return C1_B, E1_B, C2_B
"""
function up_top_B(envs::ExcEnvTensor, ky, Pt, Ptd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_B, Es_B, T_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])

    newC1_B, newE1_B, newC2_B = proj_top_B(ky, Pt, Ptd, Cs_B[1], Es[4], Cs[1], Es_B[4], 
                                                        Es_B[1], T,     Es[1], T_B, 
                                                        Cs_B[2], Es[2], Cs[2], Es_B[2]) # XXX: unnecessay computation

    env2 = EnvTensor(T_B, [newC1_B, newC2_B, Cs_B[3], Cs_B[4]], [newE1_B, Es_B[2], Es_B[3], Es_B[4]], get_maxchi(envs[2]))
    env2
end

"""
    up_top_Bd

Return C1_Bd, E1_Bd, C2_Bd
"""
function up_top_Bd(envs::ExcEnvTensor, ky, Pt, Ptd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_Bd, Es_Bd, T_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])

    newC1_Bd, newE1_Bd, newC2_Bd = proj_top_Bd(ky, Pt, Ptd, Cs_Bd[1], Es[4], Cs[1], Es_Bd[4], 
                                                            Es_Bd[1], T,     Es[1], T_Bd, 
                                                            Cs_Bd[2], Es[2], Cs[2], Es_Bd[2]) # XXX: unnecessay computation

    env3 = EnvTensor(T_Bd, [newC1_Bd, newC2_Bd, Cs_Bd[3], Cs_Bd[4]], [newE1_Bd, Es_Bd[2], Es_Bd[3], Es_Bd[4]], get_maxchi(envs[3]))
    env3
end

"""
    up_top_BB

Return C1_BB, E1_BB, C2_BB
"""
function up_top_BB(envs::ExcEnvTensor, Pt, Ptd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_B, Es_B, T_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])
    Cs_Bd, Es_Bd, T_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])
    Cs_BB, Es_BB, T_BB = corner(envs[4]), edge(envs[4]), bulk(envs[4])

    newC1_BB, newE1_BB, newC2_BB = proj_top_BB(Pt, Ptd, 
                                    Cs_BB[1],Es[4], Cs[1],Es_BB[4], Cs_B[1],Es_Bd[4], Cs_Bd[1], Es_B[4],
                                    Es_BB[1],T,     Es[1],T_BB,     Es_B[1],T_Bd,     Es_Bd[1], T_B,
                                    Cs_BB[2],Es[2], Cs[2],Es_BB[2], Cs_B[2],Es_Bd[2], Cs_Bd[2], Es_B[2]) 
    
    env4 = EnvTensor(T_BB, [newC1_BB, newC2_BB, Cs_BB[3], Cs_BB[4]], [newE1_BB, Es_BB[2], Es_BB[3], Es_BB[4]], get_maxchi(envs[4]))
    env4
end

"""
    up_bottom_B

Return C4_B, E3_B, C3_B
"""
function up_bottom_B(envs::ExcEnvTensor, ky, Pb, Pbd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_B, Es_B, T_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])

    newC4_B, newE3_B, newC3_B = proj_bottom_B(ky, Pb, Pbd, Cs_B[4], Es[4], Cs[4], Es_B[4], 
                                                           Es_B[3], T,     Es[3], T_B, 
                                                           Cs_B[3], Es[2], Cs[3], Es_B[2]) # XXX: unnecessay computation

    env2 = EnvTensor(T_B, [Cs_B[1], Cs_B[2], newC3_B, newC4_B], [Es_B[1], Es_B[2], newE3_B, Es_B[4]], get_maxchi(envs[2]))
    env2
end

"""
    up_bottom_Bd

Return C4_Bd, E3_Bd, C3_Bd
"""
function up_bottom_Bd(envs::ExcEnvTensor, ky, Pb, Pbd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_Bd, Es_Bd, T_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])

    newC4_Bd, newE3_Bd, newC3_Bd = proj_bottom_Bd(ky, Pb, Pbd, Cs_Bd[4], Es[4], Cs[4], Es_Bd[4], 
                                                               Es_Bd[3], T,     Es[3], T_Bd, 
                                                               Cs_Bd[3], Es[2], Cs[3], Es_Bd[2]) # XXX: unnecessay computation

    env3 = EnvTensor(T_Bd, [Cs_Bd[1], Cs_Bd[2], newC3_Bd, newC4_Bd], [Es_Bd[1], Es_Bd[2], newE3_Bd, Es_Bd[4]], get_maxchi(envs[3]))
    env3
end

"""
    up_bottom_BB

Return C4_BB, E3_BB, C3_BB
"""
function up_bottom_BB(envs::ExcEnvTensor, Pb, Pbd)
    Cs, Es, T = corner(envs[1]), edge(envs[1]), bulk(envs[1])
    Cs_B, Es_B, T_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])
    Cs_Bd, Es_Bd, T_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])
    Cs_BB, Es_BB, T_BB = corner(envs[4]), edge(envs[4]), bulk(envs[4])

    newC4_BB, newE3_BB, newC3_BB = proj_bottom_BB(Pb, Pbd, 
                                    Cs_BB[4],Es[4], Cs[4],Es_BB[4], Cs_B[4],Es_Bd[4], Cs_Bd[4], Es_B[4],
                                    Es_BB[3],T,     Es[3],T_BB,     Es_B[3],T_Bd,     Es_Bd[3], T_B,
                                    Cs_BB[3],Es[2], Cs[3],Es_BB[2], Cs_B[3],Es_Bd[2], Cs_Bd[3], Es_B[2]) 
    
    env4 = EnvTensor(T_BB, [Cs_BB[1], Cs_BB[2], newC3_BB, newC4_BB], [Es_BB[1], Es_BB[2], newE3_BB, Es_BB[4]], get_maxchi(envs[4]))
    env4
end