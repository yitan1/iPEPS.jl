using iPEPS

using LinearAlgebra

d = 2
D = 3
A = iPEPS.IPEPS(rand(d,D,D,D,D))

chi = 10
env0 = iPEPS.init_env(A, chi);

env, s = iPEPS.up_left!(env0, A);

corner