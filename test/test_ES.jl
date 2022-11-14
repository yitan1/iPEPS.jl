using iPEPS

using LinearAlgebra

using BenchmarkTools

d = 2
D = 3
phi = iPEPS.IPEPS(rand(d,D,D,D,D));
A = iPEPS.data(phi);

chi = 10
env = iPEPS.get_envtensor(phi; chi = chi);

dA = iPEPS.get_norm_dA(env, phi);

# @btime dA = iPEPS.get_norm_dA1($env, $phi);

dA = reshape(dA, (1,:) );
basis = nullspace(dA);
size(basis)


B1 = reshape(basis[:,3], d,D,D,D,D);
T = iPEPS.transfer_matrix(A, B1);
iPEPS.get_norm(env, T)