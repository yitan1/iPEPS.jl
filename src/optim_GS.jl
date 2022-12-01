export optimize_GS

function optimize_GS(A, h_hor, h_ver; chi = 30)
    f(x) = get_energy(x, h_hor, h_ver; chi = chi) 
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
    T = transfer_matrix(A)
    rho = density_matrix(A)
    env = get_envtensor(T; chi = chi, output = false)

    E_hor, N_hor = get_hor_E_N(h_hor, env, rho, T)
    E_ver, N_ver = get_ver_E_N(h_ver, env, rho, T)

    E0 = E_hor/N_hor + E_ver/N_ver
    @show E0/2
    E0/2
end

function get_hor_E_N(h_hor, env::EnvTensor, rho, T) #TODO
    Cs, Es = corner(env), edge(env)

    effT = contract_hor_T(T, T)
    effTh = contract_hor_Th(h_hor, rho, rho)
    effE1 = contract_E1(Es[1], Es[1])
    effE3 = contract_E3(Es[3], Es[3])

    E_hor = contract_env(Cs[1], Cs[2], Cs[3], Cs[4], effE1, Es[2], effE3, Es[4], effTh)
    N_hor = contract_env(Cs[1], Cs[2], Cs[3], Cs[4], effE1, Es[2], effE3, Es[4], effT)

    E_hor, N_hor
end

function get_ver_E_N(h_ver, env::EnvTensor, rho, T)
    Cs, Es = corner(env), edge(env)

    effT = contract_ver_T(T, T)
    effTh = contract_ver_Th(h_ver, rho, rho)
    effE2 = contract_E2(Es[2], Es[2])
    effE4 = contract_E4(Es[4], Es[4])

    E_ver = contract_env(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], effE2, Es[3], effE4, effTh)
    N_ver = contract_env(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], effE2, Es[3], effE4, effT)

    E_ver, N_ver
end
