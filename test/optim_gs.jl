using PhyOperators
using iPEPS
using MKL
using Optim
using LinearAlgebra
using Zygote

SI = op("SI", "Spinhalf") 
Sx = op("Sx", "Spinhalf")
Sy = op("Sy", "Spinhalf")
Sz = op("Sz", "Spinhalf")

hh = 1*kron(kron(SI, Sx), kron(Sx, SI)) + ( kron(kron(Sz, Sz), kron(SI, SI) )  + kron(kron(SI, SI), kron(Sz, Sz)) ) /2  .|> real
hv = 1*kron(kron(SI, Sy), kron(Sy, SI)) + ( kron(kron(Sz, Sz), kron(SI, SI) )  + kron(kron(SI, SI), kron(Sz, Sz)) ) /2  .|> real
H = [-hh, -hv] 

D = 2
d = 4
A = randn(D,D,D,D,d)
A = A/maximum(abs,A);
e0 = iPEPS.run(H, A)

gradient(x -> iPEPS.run(H, x), A)

ts0 = iPEPS.CTMTensors(A,A);
ts, s = iPEPS.run_ctm(ts0, 50);



