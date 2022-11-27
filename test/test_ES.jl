using iPEPS
using LinearAlgebra
using BenchmarkTools
using Zygote

####
d = 2
D = 2
A = rand(d,D,D,D,D)
Ad = conj(A)
phi = iPEPS.IPEPS(A);
####

###
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

g = gradient(x -> f(A, x), Ad);

@btime f($A, $Ad)
@btime f1($A, $Ad)
####

####
chi = 5
env = iPEPS.get_envtensor(phi; chi = chi);
dA = iPEPS.get_norm_dA(env, phi);
# @btime dA = iPEPS.get_norm_dA1($env, $phi);

####
dA = reshape(dA, (1,:) );
basis = nullspace(dA);
size(basis)
B1 = reshape(basis[:,3], d,D,D,D,D);
T = iPEPS.transfer_matrix(A, B1);
iPEPS.get_norm(env, T) 

g = gradient(_x -> iPEPS.get_norm(env, iPEPS.transfer_matrix(_x, _x)), A)[1];
####


####
d = 2
D = 2
chi = 5
A = rand(d,D,D,D,D);
B = rand(d,D,D,D,D);
Bd = rand(d,D,D,D,D);
phi1 = iPEPS.ExcIPEPS(A,B);
phi2 = iPEPS.ExcIPEPS(A,Bd);

envs0 = iPEPS.init_env(phi1, phi2, chi);

env, s = iPEPS.update_env(envs0, 0, 0);

envs = iPEPS.get_envtensor(phi1, phi2);




##########
using PhyOperators
h = spinmodel();
h = reshape(h, 2,2,2,2);
d = 2
D = 2
using MKL

A = rand(d,D,D,D,D);
phi0 = iPEPS.IPEPS(A);
Bn = iPEPS.get_tangent_basis(phi0; chi = 5);

H, N = iPEPS.eff_hamitonian_norm(h, h, 0.0, 0.0, phi0, Bn; chi = 5);


Bj = Bn[:,1];
Bdj = conj(Bj);

@time dE = gradient(_x -> iPEPS.effH_ij(h, h, 0.0, 0.0, phi0, Bj, _x, 5), Bdj)[1];
@time dN = gradient(_x -> iPEPS.effN_ij(0.0, 0.0, phi0, Bj, _x, 5), Bdj)[1];

@time dEN = jacobian(_x -> iPEPS.effH_N_ij(h, h, 0.0, 0.0, phi0, Bj, _x, 5), Bdj)[1];


@time iPEPS.effH_ij(h, h, 1.0, 0.0, phi0, Bj, Bdj, 5)
@time iPEPS.effN_ij(1.0, 0.0, phi0, Bj, Bdj, 5)
@time iPEPS.effH_N_ij(h, h, 0.0, 0.0, phi0, Bj, Bdj, 5)


d = 2
id = Matrix(I, d,d)
hI = kron(id,id)
hI = reshape(hI, d, d, d, d);
@time iPEPS.effH_ij(hI, hI, 0.0, 0.0, phi0, Bj, Bdj, 5)