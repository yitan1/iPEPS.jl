using iPEPS
using LinearAlgebra
using BenchmarkTools
using Zygote
using MKL

#### Ground state
d = 2
D = 2
A = rand(d,D,D,D,D)
Ad = conj(A)
phi = iPEPS.IPEPS(A);
####

#### comparison time about whether inplace or not  
function f(A,Ad)
    phi = iPEPS.IPEPS(A, Ad)
    env = iPEPS.get_envtensor(phi; chi = 5, output = false)
    iPEPS.get_norm(env) 
end
function f1(A,Ad)
    phi = iPEPS.IPEPS(A, Ad)
    env = iPEPS.get_envtensor(phi; chi = 5, inplace = true, output = false)
    iPEPS.get_norm(env) 
end

# g = gradient(x -> f(A, x), Ad);

@btime f($A, $Ad)
@btime f1($A, $Ad)
####

#### Test autodiff
chi = 5
env = iPEPS.get_envtensor(phi; chi = chi);
@btime dA = iPEPS.get_norm_dA($env, $phi);
@btime dA1 = iPEPS.get_norm_dA1($env, $phi);


#### excited basis
using PhyOperators
h = spinmodel();
h = reshape(h, 2,2,2,2);
d = 2
D = 2

A = rand(d,D,D,D,D);
phi0 = iPEPS.IPEPS(A);
Bn = iPEPS.get_tangent_basis(phi0; chi = 5);
#### 

#### test eigenvalue
H, N = iPEPS.eff_hamitonian_norm(h, h, 0.0, 0.0, phi0, Bn; chi = 5);
H = (H + H')/2
N = (N + N')/2
isposdef(N)
ishermitian(H)
F= eigen(H, N)
####


#### comparison time about autodiff H,N and HN  
Bj = Bn[:,1];
Bdj = conj(Bj);
@time dE = gradient(_x -> iPEPS.effH_ij(h, h, 0.0, 0.0, phi0, Bj, _x, 5), Bdj)[1];
@time dN = gradient(_x -> iPEPS.effN_ij(0.0, 0.0, phi0, Bj, _x, 5), Bdj)[1];

@time dEN = jacobian(_x -> iPEPS.effH_N_ij(h, h, 0.0, 0.0, phi0, Bj, _x, 5), Bdj)[1];
#####

#### test 
@time iPEPS.effH_ij(h, h, 1.0, 0.0, phi0, Bj, Bdj, 5)
@time iPEPS.effN_ij(1.0, 0.0, phi0, Bj, Bdj, 5)
@time iPEPS.effH_N_ij(h, h, 0.0, 0.0, phi0, Bj, Bdj, 5)
#### 

#### test norm speed with I
d = 2
id = Matrix(I, d,d)
hI = kron(id,id)
hI = reshape(hI, d, d, d, d);
@time iPEPS.effH_ij(hI, hI, 0.0, 0.0, phi0, Bj, Bdj, 5)
####