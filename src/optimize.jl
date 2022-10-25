using Zygote, Optim

function optimize_GS(A, h; chi = 30)
    f(x) = forward(x, h; chi = chi) 
    function fg!(F,G,x)
        y, back = Zygote.pullback(f, x)
        if G != nothing
            copy!(G, back(1)[1])
        end
        if F != nothing
            return y
        end
    end

    res = optimize(Optim.only_fg!(fg!), A, LBFGS(), Optim.Options(x_tol = 1e-9, f_tol = 1e-9, g_tol = 1e-7))
    res
end

# function optimize_GS(A, h; chi = 30)
#     f(x) = forward(x, h; chi = chi) 

#     g(x) = gradient(f, x)[1]
#     g!(G,x) = copy!(G, g(x))

#     res = optimize(f, g!, A, LBFGS(), Optim.Options(x_tol = 1e-9, f_tol = 1e-9, g_tol = 1e-5))
#     res
# end

function forward(A, h; chi = 30)
    C,E = CTM(A; chi= chi)
    e0 = op_expect(A ,C, E, h)
    @show e0
    e0
end