using Zygote, Optim

function sym_optimize_GS(A, h; chi = 30)
    f(x) = forward(x, h; chi = chi) 
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

# function optimize_GS(A, h; chi = 30)
#     f(x) = forward(x, h; chi = chi) 

#     g(x) = gradient(f, x)[1]
#     g!(G,x) = copy!(G, g(x))

#     res = optimize(f, g!, A, LBFGS(), Optim.Options(x_tol = 1e-9, f_tol = 1e-9, g_tol = 1e-5))
#     res
# end

function forward(A, h; chi = 30)
    A = symmetrize(A)
    C,E = CTM(A; chi= chi)
    e0 = op_expect(A ,C, E, h)
    @show e0
    e0
end


function symmetrize(A)
    # A(phy, up, left, down, right)
    # left-right, up-down, diagonal symmetrize
    Asymm = (A + permutedims(A, (1, 2, 5, 4, 3)))/2           # left-right symmetry
    Asymm = (Asymm + permutedims(Asymm, (1, 4, 3, 2, 5)))/2   # up-down symmetry
    Asymm = (Asymm + permutedims(Asymm, (1, 5, 4, 3, 2)))/2   # skew-diagonal symmetry
    Asymm = (Asymm + permutedims(Asymm, (1, 3, 2, 5, 4)))/2   # diagonal symmetry

    return Asymm/norm(Asymm)

end


# ############
# using Flux
# using Zygote, Optim
# using KrylovKit
# function loss(A, h)
#     C,E = CTM(A)
#     e0 = op_expect(A ,C, E, h)
#     # @show e0
#     e0
# end

# grad_loss(A) = gradient(_x -> loss(_x, h), A)[1][:]

# delta = 1e-5
# iter = 100
# for i = 1:iter
#     e, A0 = eigsolve(grad_loss, A[:], 1, :LR, Lanczos())
#     e0 = e[1]
#     tol = norm(A[:] - A0[1])
#     @show e0,tol
#     if tol < delta || i == iter
#         @show i, tol
#         break
#     end
#     # A = convert.(Float64, A0[1])
#     A = real(A0[1])
#     # @show A
# end


# ############
# A1, e0 = optimize_GS!(h,A,100)

# function optimize_GS!(h, A, iter = 500)
#     delta = 1e-10
#     oldA = deepcopy(A)
#     for i = 1:iter

#         function loss(h)
#             C,E = CTM(A)
#             e0 = op_expect(A ,C, E, h)
#             @show e0
#             e0
#         end

#         p = Flux.params(A)
#         grads = gradient(() -> loss(h), p)

#         opt = Descent()
#         Flux.Optimise.update!(opt, p, grads)

#         tol = norm(oldA - A)/length(A)
#         if tol < delta || i == iter
#             @show i, tol
#             break
#         end
#         oldA = deepcopy(A)
#     end
#     e0 = loss(h)
#     A, e0
# end