
function optimize_ES(A, h_hor, h_ver; chi = 30)
    f(x) = forward(x, h_hor, h_ver; chi = chi) 
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

function forward(A, h_hor, h_ver; chi=30)
    T = transfer_matrix(A)
    rho = density_matrix(A)
    env = get_envtensor(T; chi = chi)
    e_hor = get_hor_energy(rho, h_hor, env)/get_hor_norm(env) 
    e_ver = get_ver_energy(rho, h_ver, env)/get_ver_norm(env)
    e0 = e_hor + e_ver
    
    @show e0
    e0
end

function get_hor_energy(rho, h_hor, env::EnvTensor) #TODO
    
end

function get_hor_norm(env::EnvTensor)
    
end

function get_ver_energy(rho, h_ver, env::EnvTensor)
    
end

function get_ver_norm(env::EnvTensor)
    
end