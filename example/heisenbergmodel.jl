using PhyOperators
using iPEPS
using MKL
using Optim
using LinearAlgebra

SI = op("SI", "Spinhalf") 
Sx = op("Sx", "Spinhalf")
Sy = op("Sy", "Spinhalf")
Sz = op("Sz", "Spinhalf")

# h = rotate_heisenberg()
# ht = reshape(h, 2, 2, 2, 2)

h = spinmodel((-1,0,0), (0,0,-3))
ht = reshape(h, 2, 2, 2, 2) |> real

d = size(h,1) |> sqrt |> Int
D = 2
chi = 30
A = rand(d,D,D,D,D)
A /= norm(A);

res = sym_optimize_GS(A,h; chi = chi)
res2 = optimize_GS(A, ht, ht; chi = chi)

@show res

A1 = Optim.minimizer(res2);
using JLD2
jldsave("gs_ising_D2_chi30.jld2"; A = A1)

iPEPS.forward(A, h)

using NPZ
A = npzread("example/jl_hei_D2_X40.npy")
A = permutedims(A, (1, 3, 4, 5, 2))

A1 = load("example/gs_ising_D2_chi30.jld2")["A"]
sym_A1 = iPEPS.symmetrize(A1)
iPEPS.get_energy(sym_A1, ht, ht; chi = 40)

phi = iPEPS.IPEPS(sym_A1);
iPEPS.effH_ij(ht, ht, 3., 2., phi, sym_A1, sym_A1, 40)/iPEPS.effN_ij(3., 2., phi, sym_A1, sym_A1, 40)

EN0 = iPEPS.effH_N_ij(ht, ht, 3., 2., phi, sym_A1, sym_A1, 40)
EN0[1]/EN0[2]

using Zygote
T = iPEPS.transfer_matrix(A);
env0 = iPEPS.get_envtensor(T; chi = chi, output = false);
f(x) = iPEPS.get_energy(A, x, ht, ht, env0; chi = 30);
# f(x) = iPEPS.forward(x, h; chi = 30);
grad = gradient(f, A)[1]



function test(x1, x2)
    v_1 = Zygote.@showgrad x1
    v0 = Zygote.@showgrad x2
    v1 = Zygote.@showgrad log(v_1)
    v2 = Zygote.@showgrad v_1*v0 
    v3 = sin(v0)
    v4 = v1 + v2
    v5 = v4 - v3
    v5
end
gradient(test, 2,5)

R1 = rand(100,100)
R2 = rand(100,100);
function f2(T, R2)
    R1 = T
    R2 = permutedims(R2,(2,1))
    P1, P2, S1 = iPEPS.get_projector(R1,R2,30)
    P2[1]
end
gradient(x -> f2(R1, x), R2)

T1 = iPEPS.transfer_matrix(rand(2,2,2,2,2));
T2 =  iPEPS.transfer_matrix(rand(2,2,2,2,2));
env0 = iPEPS.get_envtensor(T1; chi = chi, output = false);
Cs = iPEPS.corner(env0)
Es = iPEPS.edge(env0)
C4, E3, E4 = Cs[4], Es[3], Es[4];
env1 = iPEPS.get_envtensor(T2; chi = chi, output = false);
Cs1 = iPEPS.corner(env1)
Es1 = iPEPS.edge(env1)
C13, E13, E12 = Cs1[3], Es1[3], Es1[2];

using TensorOperations, TensorRules
# function f1(T1, T2, env0, env1)
@∇ function f1(T1, T2, C4,E3,E4, C13,E13,E12)
    # Cs = iPEPS.corner(env0)
    # Es = iPEPS.edge(env0)
    # env0 = iPEPS.EnvTensor(T1, Cs, Es, iPEPS.get_maxchi(env0))

    # BL = iPEPS.contract_bl_env(Cs[4],Es[3],Es[4],T1)
    BL = iPEPS.contract_bl_env(C4,E3,E4,T1)
    # BL = rand(Float64, 120,120)

    # Cs1 = iPEPS.corner(env1)
    # Es1 = iPEPS.edge(env1)
    # env1 = iPEPS.EnvTensor(T2, Cs1, Es1, iPEPS.get_maxchi(env1))

    # BR = iPEPS.contract_br_env(Cs1[3],Es1[3],Es1[2],T2)
    # BR = iPEPS.contract_br_env(C13,E13,E12,T2)
    # BR = rand(Float64, 120,120)
    @tensor BR[m1,m2,m3,m4] := C13[p2,p1]*E13[p3,m3,p1]*E12[m1,p4,p2]*T2[m2,m4,p3,p4]
    BR = reshape(BR, size(BR,1)*size(BR,2), :)
    R1 = BL
    R2 = permutedims(BR,(2,1))

    chi = 30
    new_chi = min(chi, 120)
    W = R1*R2
    U, S, V = svd(W)
    S[1]
    U1 = U[:, 1:new_chi]
    V1 = V[:, 1:new_chi]
    # S = S./S[1]
    S1 = S[1:new_chi]
    
    # cut_off = sum(S[new_chi+1:end]) / sum(S)   

    inv_sqrt_S = sqrt.(S1) |> diagm |> inv

    P1 = R2*V1*inv_sqrt_S    #(D2,D3)*(D3,chi)*(chi,chi) --> (D2, chi) 
    P2 = inv_sqrt_S*U1'*R1   #(chi,chi)*(chi,D1)*(D1,D2)  --> (chi, D2)
    P2[1]
end
gradient(x -> f1(x, T2, C4,E3,E4, C13,E13,E12), T1)





# C, E = CTM(A1; chi = 30);
# Mx = kron(Sx,SI)
# Mz = kron(Sz,SI)
# My = kron(Sy,SI)
# op_expect(A1 ,C, E, h)