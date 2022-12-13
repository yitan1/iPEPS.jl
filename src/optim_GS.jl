export optimize_GS

function optimize_GS(A, h_hor, h_ver; chi = 30)
    f(x) = get_energy(x, conj(x), h_hor, h_ver; chi = chi) 
    function fg!(F,G,x)
        y, back = Zygote.pullback(f, x)
        if G !== nothing
            copy!(G, back(1)[1])
        end
        if F !== nothing
            return y
        end
    end

    res = optimize(Optim.only_fg!(fg!), A, LBFGS(), Optim.Options(x_tol = 1e-9, f_tol = 1e-9, g_tol = 1e-7))
    res
end

function get_energy(A, h_hor, h_ver; chi=30)
    get_energy(A, conj(A), h_hor, h_ver; chi=chi)
end
function get_energy(A, Ad, h_hor, h_ver; chi=30)
    T = transfer_matrix(A, Ad)
    rho = density_matrix(A, Ad)
    env = get_envtensor(T; chi = chi, output = false)

    E_hor, N_hor = get_hor_E_N(h_hor, env, rho)
    E_ver, N_ver = get_ver_E_N(h_ver, env, rho)
    
    E0 =  (E_hor + E_ver)/2
    # E0 = (E_hor + E_ver)/2
    @show E0, E_hor, E_ver, N_hor, N_ver
    E0
end

function get_hor_E_N(h_hor, env::EnvTensor, rho) #TODO
    Cs, Es = corner(env), edge(env)

    # effT = contract_hor_T(T, T)
    # effTh = contract_hor_Th(h_hor, rho, rho)
    # effE1 = contract_E1(Es[1], Es[1])
    # effE3 = contract_E3(Es[3], Es[3])

    rho = get_rho_hor(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], Es[1], Es[2], Es[3], Es[3], Es[4], rho, rho)

    rrho = reshape(rho, size(rho,1)*size(rho,2), :)
    N_hor = tr(rrho)

    hh = reshape(h_hor, size(h_hor,1)*size(h_hor,2), :)
    E_hor = tr(hh*rrho)/N_hor

    # E_hor = contract_env(Cs[1], Cs[2], Cs[3], Cs[4], effE1, Es[2], effE3, Es[4], effTh)
    # N_hor = contract_env(Cs[1], Cs[2], Cs[3], Cs[4], effE1, Es[2], effE3, Es[4], effT)

    E_hor, N_hor
end

function get_ver_E_N(h_ver, env::EnvTensor, rho)
    Cs, Es = corner(env), edge(env)

    # effT = contract_ver_T(T, T)
    # effTh = contract_ver_Th(h_ver, rho, rho)
    # effE2 = contract_E2(Es[2], Es[2])
    # effE4 = contract_E4(Es[4], Es[4])

    rho = get_rho_ver(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], Es[2], Es[2], Es[3], Es[4], Es[4], rho, rho)

    rrho = reshape(rho, size(rho,1)*size(rho,2), :)
    N_ver = tr(rrho)

    # E_ver = contract_env(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], effE2, Es[3], effE4, effTh)
    # N_ver = contract_env(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], effE2, Es[3], effE4, effT)

    hh = reshape(h_ver, size(h_ver,1)*size(h_ver,2), :)
    E_ver = tr(hh*rrho)/N_ver

    E_ver, N_ver
end
