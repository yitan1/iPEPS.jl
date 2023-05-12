using PhyOperators
using iPEPS
# using MKL
# using Optim
using LinearAlgebra
using Zygote
using NPZ
using ProfileView 

SI = op("SI", "Spinhalf") 
Sx = op("Sx", "Spinhalf")
Sy = op("Sy", "Spinhalf")
Sz = op("Sz", "Spinhalf")

hh = 1*kron(kron(SI, Sx), kron(Sx, SI)) .+ ( kron(kron(Sz, Sz), kron(SI, SI) ) .+ kron(kron(SI, SI), kron(Sz, Sz)) ) /2  .|> real
hv = 1*kron(kron(SI, Sy), kron(Sy, SI)) .+ ( kron(kron(Sz, Sz), kron(SI, SI) )  .+ kron(kron(SI, SI), kron(Sz, Sz)) ) /2  .|> real
H = [-hh, -hv] 

h = -kron(Sx, Sx) .- 2* kron(Sz, SI) / 2 .- 2*kron(SI, Sz) / 2
H = [h, h] 

using TensorOperations
function init_A()
     Q_op = zeros(ComplexF64,2,2,2,2,2)
     Q_op[:,:,1,1,1] = SI
     Q_op[:,:,1,2,2] = Sx
     Q_op[:,:,2,1,2] = Sy
     Q_op[:,:,2,2,1] = Sz
     ux, uy, uz = 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)
     s111 = 1/sqrt(2+2*uz)*[1 + uz, ux + im*uy]

     @tensor l[-1,-2,-3,-4] := Q_op[-1,1,-2,-3,-4]*s111[1]
     @tensor r[-1,-2,-3,-4] := Q_op[-1,1,-4,-3,-2]*s111[1]
     @tensor A[-1,-2,-3,-4,-5,-6] := l[-1,-3,-4, 1]*r[-2, 1, -5,-6]
     dimA = size(A)
     A = reshape(A, dimA[1]*dimA[2], dimA[3], dimA[4], dimA[5], dimA[6])
     A = permutedims(A, (2,3,4,5,1))
end
A = init_A() |> real;

D = 2
d = 4
A = randn(D,D,D,D,d);
A = A ./ maximum(abs,A);

A = npzread("honey_015_A.npz")["A"] 
A = dropdims(A, dims = 1)
A = permutedims(A, (3,4,5,2,1));

ts0 = iPEPS.CTMTensors(A,A);
function conv_fun(_x)
     E, N = iPEPS.get_energy(H,_x)
     E[1] + E[2]
end
ts0, s = iPEPS.run_ctm(ts0, 50, conv_fun = conv_fun);
ProfileView.@profview iPEPS.run_energy(H, ts0, 50, A)

ProfileView.@profview gradient(x -> iPEPS.run_energy(H, ts0, 50, x), A)

iPEPS.optim_GS(H, A, 50)
 
 using ProfileView
 ProfileView.@profview profile_test(10)

 