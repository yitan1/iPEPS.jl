using iPEPS
using Test

# @testset "iPEPS.jl" begin
#     # Write your tests here.
# end

using LinearAlgebra
using PhyOperators
using iPEPS
using MKL

SI = op("SI", "Spinhalf")
Sx = op("Sx", "Spinhalf")
Sy = op("Sy", "Spinhalf")
Sz = op("Sz", "Spinhalf")
Sm = op("Sm", "Spinhalf")
Sp = op("Sp", "Spinhalf")

# h0 = kron(Sz,Sz,SI,SI) + kron(Sz,SI,Sz,SI) + kron(SI,Sz,SI,Sz) + kron(SI,SI,Sz,Sz) + 
#         kron(Sx,Sx,SI,SI) + kron(Sx,SI,Sx,SI) + kron(SI,Sx,SI,Sx) + kron(SI,SI,Sx,Sx) + 
#             kron(Sy,Sy,SI,SI) + kron(Sy,SI,Sy,SI) + kron(SI,Sy,SI,Sy) + kron(SI,SI,Sy,Sy)
# h0 = real(h0) 

h1 = 2*kron(Sz,4*Sx'*Sz*Sx) - kron(Sm,4*Sx'*Sp*Sx) - kron(Sp,4*Sx'*Sm*Sx)

h2 = 2*kron(Sz,4*Sx'*Sz*Sx) - 2*kron(Sx,4*Sx'*Sx*Sx) - 2*kron(Sy,4*Sx'*Sy*Sx) |> real


h = real(spinmodel())
permutedims(reshape(h,(2,2,2,2)), (1,3,2,4))

d = size(h2,1) |> sqrt |> Int 
D = 2
chi = 30
A = rand(d,D,D,D,D)
A = A/norm(A);

res = sym_optimize_GS(A,h2; chi)

using Optim
A1 = Optim.minimizer(res); 


C, E = CTM(A1; chi = 30);
Mx = kron(Sx,SI)
Mz = kron(Sz,SI)
My = kron(Sy,SI)
op_expect(A1 ,C, E, My)

@show res





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

