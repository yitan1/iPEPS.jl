using iPEPS

using LinearAlgebra

d = 2
D = 3
A = iPEPS.IPEPS(rand(d,D,D,D,D));

chi = 10
env = iPEPS.get_envtensor(A; chi = chi);