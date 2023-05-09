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

hh = 0.2*kron(kron(SI, Sx), kron(Sx, SI)) .+ ( kron(kron(Sz, Sz), kron(SI, SI) ) .+ kron(kron(SI, SI), kron(Sz, Sz)) ) /2  .|> real
hv = 0.2*kron(kron(SI, Sy), kron(Sy, SI)) .+ ( kron(kron(Sz, Sz), kron(SI, SI) )  .+ kron(kron(SI, SI), kron(Sz, Sz)) ) /2  .|> real
H = [-hh, -hv] 

D = 2
d = 4
A = randn(D,D,D,D,d)
A = A ./ maximum(abs,A);
ts0 = iPEPS.CTMTensors(A,A);
ts0, s = iPEPS.run_ctm(ts0, 30);
e0 = iPEPS.run_energy(H, ts0, A)

gradient(x -> iPEPS.run(H, ts0, x), A)[1]

iPEPS.optim_GS(H, A)



