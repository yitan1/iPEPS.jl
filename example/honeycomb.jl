using PhyOperators
using iPEPS
using MKL
using Optim
using LinearAlgebra

SI = op("SI", "Spinhalf") 
Sx = op("Sx", "Spinhalf")
Sy = op("Sy", "Spinhalf")
Sz = op("Sz", "Spinhalf")

hh = kron(kron(I, Sx), kron(Sx, I)) + ( kron(kron(Sz, Sz), kron(I, I) )  + kron(kron(I, I), kron(Sz, Sz)) ) /2
hv = kron(kron(I, Sy), kron(Sy, I)) + ( kron(kron(Sz, Sz), kron(I, I) )  + kron(kron(I, I), kron(Sz, Sz)) ) /2

d = 4
D = 2
chi = 30
A = rand(d,D,D,D,D)
A /= norm(A);

res = optimize_GS(A, hh, hv; chi = chi)