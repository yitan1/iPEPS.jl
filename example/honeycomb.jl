using PhyOperators
using iPEPS
using MKL
using Optim
using LinearAlgebra

SI = op("SI", "Spinhalf") 
Sx = op("Sx", "Spinhalf")
Sy = op("Sy", "Spinhalf")
Sz = op("Sz", "Spinhalf")

hh = 0.2*kron(kron(SI, Sx), kron(Sx, SI)) + ( kron(kron(Sz, Sz), kron(SI, SI) )  + kron(kron(SI, SI), kron(Sz, Sz)) ) /2
hv = 0.2*kron(kron(SI, Sy), kron(Sy, SI)) + ( kron(kron(Sz, Sz), kron(SI, SI) )  + kron(kron(SI, SI), kron(Sz, Sz)) ) /2
hh = reshape(hh, 4,4,4,4)
hv = reshape(hv, 4,4,4,4)

h = rotate_heisenberg()
hh = reshape(h, 2, 2, 2, 2)
hv = reshape(h, 2, 2, 2, 2)

d = 2
D = 2
chi = 30
A = rand(d,D,D,D,D)
A /= norm(A);

res = optimize_GS(A, hh, hv; chi = chi)