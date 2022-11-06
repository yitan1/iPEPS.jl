using iPEPS

using LinearAlgebra

d = 2
D = 3
A = iPEPS.IPEPS(rand(d,D,D,D,D))

env0 = iPEPS.init_env(A);

A = rand(30,20)
B = rand(20,10)
chi = 15

P1,P2 = iPEPS.get_projector(A,B,chi)