using PhyOperators
using iPEPS
using MKL
using Optim
using LinearAlgebra

SI = op("SI", "Spinhalf") 
Sx = op("Sx", "Spinhalf")
Sy = op("Sy", "Spinhalf")
Sz = op("Sz", "Spinhalf")

Q_op = zeros(ComplexF64,2,2,2,2,2)
Q_op[:,:,1,1,1] = SI
Q_op[:,:,1,2,2] = Sx
Q_op[:,:,2,1,2] = Sy
Q_op[:,:,2,2,1] = Sz
ux, uy, uz = 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)
s111 = 1/sqrt(2+2*uz)*[1 + uz, ux + im*uy]

using TensorOperations
@tensor l[-1,-2,-3,-4] := Q_op[-1,1,-2,-3,-4]*s111[1]
@tensor r[-1,-2,-3,-4] := Q_op[-1,1,-4,-3,-2]*s111[1]
@tensor A[-1,-2,-3,-4,-5,-6] := l[-1,-3,-4, 1]*r[-2, 1, -5,-6]
dimA = size(A)
A = reshape(A, dimA[1]*dimA[2], dimA[3], dimA[4], dimA[5], dimA[6]);

phi = 0.24*pi
a = tan(phi)
R_op = zeros(ComplexF64,2,2,2,2,2)
R_op[:,:,1,1,1] = SI
R_op[:,:,1,2,2] = Sx*a
R_op[:,:,2,1,2] = Sy*a
R_op[:,:,2,2,1] = Sz*a

@tensor RR[:] := R_op[-1,-3,-5,-6,1]*R_op[-2,-4,-8,-7,1]
dRR = size(RR)
RR = reshape(RR, dRR[1]*dRR[2], dRR[3]*dRR[4], dRR[5], dRR[6], dRR[7], dRR[8])
@tensor A1[:] := RR[-1,1, -2, -4, -6, -8]*A[1, -3, -5, -7, -9]
D1 = size(A1,2)
D2 = size(A1,3)
A1 = reshape(A1, size(A1,1), D1*D2, D1*D2, D1*D2, D1*D2)

hh = 1*kron(kron(SI, Sx), kron(Sx, SI)) + ( kron(kron(Sz, Sz), kron(SI, SI) )  + kron(kron(SI, SI), kron(Sz, Sz)) ) /2
hv = 1*kron(kron(SI, Sy), kron(Sy, SI)) + ( kron(kron(Sz, Sz), kron(SI, SI) )  + kron(kron(SI, SI), kron(Sz, Sz)) ) /2
hh = -reshape(hh, 4,4,4,4)
hv = -reshape(hv, 4,4,4,4)

e0 = iPEPS.get_energy(A1, hh, hv, chi = 64)

h = rotate_heisenberg()
hh = reshape(h, 2, 2, 2, 2)
hv = reshape(h, 2, 2, 2, 2)

d = 2
D = 2
chi = 30
A = rand(d,D,D,D,D)
A /= norm(A);

res = optimize_GS(A, hh, hv; chi = chi)