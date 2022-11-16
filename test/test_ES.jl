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

envs = iPEPS.init_env(phi1, phi2, chi);