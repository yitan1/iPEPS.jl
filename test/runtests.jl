using iPEPS
using Test

# @testset "iPEPS.jl" begin
#     # Write your tests here.
# end

# using TensorOperations, LinearAlgebra
# using Zygote, Optim
# using BenchmarkTools

# SI = [1 0; 0 1]
# Sx = [0 1; 1 0]/2
# Sy = [0.0 -1.0im; 1.0im 0.0]/2
# Sz = [1 0; 0 -1]/2
# Sp = [0 1; 0 0]
# Sm = [0 0; 1 0]

# h = kron(Sz,Sz) + kron(Sp,Sm)/2 + kron(Sm,Sp)/2

# d = size(h,1) |> sqrt |> Int
# D = 2
# A = rand(d,D,D,D,D);

# res = optimize_GS(A,h; chi = 30)





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

