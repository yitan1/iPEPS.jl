using iPEPS
using JLD2, TOML
# using MKL
# using Random  

# rng = MersenneTwister(4);
# D = 2
# A = randn(rng, Float64, D,D,D,D,4);
# A = iPEPS.get_symmetry(A)

A = iPEPS.init_hb_gs(2, p1 = 0.24, p2 = 0, dir = "XX");
jldsave("simulation/hb_g11_D2_X32/gs.jld2", A = A)
H = iPEPS.honeycomb(1, 1,dir = "XX");
# res = optim_gs(H, A, "")

prepare_basis(H, "")

pxs, pys = make_es_path()
n = 20
px, py = pxs[n], pys[n]
optim_es(px, py, "");

##################

H = load("simulation/hb_g11_D2_X32/basis.jld2", "H")
A = iPEPS.init_hb_gs(2)
# A = iPEPS.
cfg = TOML.parsefile("src/default_config.toml")
ts = iPEPS.CTMTensors(A, cfg);

conv_fun(_x) = iPEPS.get_gs_energy(_x, H)[1]
# conv_fun(_x) = iPEPS.get_gs_norm(_x)
ts, _ = iPEPS.run_ctm(ts, conv_fun = conv_fun);

basis = iPEPS.cut_basis(ts)
vs, vecs = iPEPS.diag_n_dm(ts)

C1, C2, C3, C4 = ts.Cs
E1, E2, E3, E4 = ts.Es
n_dm = iPEPS.get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4);

# bl = reshape(basis[:, 2], size(ts.A))
ndm_Ad = iPEPS.tcon([n_dm, ts.Ad], [[-1,-2,-3,-4,1,2,3,4], [1,2,3,4,-5]]);
ndm_A = iPEPS.tcon([n_dm, ts.A], [[1,2,3,4,-1,-2,-3,-4], [1,2,3,4,-5]]);

t = [iPEPS.tcon([ndm_Ad, reshape(basis[:, i], size(ts.A))], [[1,2,3,4,5], [1,2,3,4,5]])[1] for i in axes(basis, 2)]  .|> abs;

[iPEPS.tcon([ndm_A, reshape(basis[:, i], size(ts.A))], [[1,2,3,4,5], [1,2,3,4,5]])[1] for i in axes(basis, 2)]

transpose(basis)*ndm_Ad[:]
basis'*ndm_A[:]

s1 = iPEPS.sigmaz
s2 = iPEPS.sigmaz
op1 = iPEPS.tout(s1, s2)
SA = iPEPS.tcon([ts.A, op1], [[-1,-2,-3,-4,1], [-5,1]]);

transpose(ts.A[:])* ndm_Ad[:]

nAA = transpose(ts.A[:])*ndm_Ad[:]
bm = zeros(ComplexF64, size(basis))
for i in axes(basis, 2)
    bm[:, i] = basis[:, i] .- (transpose(basis[:,i]) * ndm_Ad[:]) * ts.A[:]./nAA
    
    # @show transpose(bm[:, i])*ndm_Ad[:]

    ndm_SA = iPEPS.tcon([n_dm, SA], [[1,2,3,4,-1,-2,-3,-4], [1,2,3,4,-5]])
    # @transpose(basis)*ndm_Ad[:]
    # @show bm[:, i]'*ndm_SA[:] 
end

t = bm'*ndm_SA[:] .|> abs 

B1 = reshape(bm[:,4], size(ts.A));

ts1 = iPEPS.normalize_gs(ts);

ts1 = setproperties(ts1, B=B1, Bd=conj(B1));
# conv_fun(_x) = iPEPS.get_all_norm(_x)[1] 
conv_fun(_x) = iPEPS.get_es_energy(_x, H) / iPEPS.get_all_norm(_x)[1]
ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);

_, envB = iPEPS.get_all_norm(ts1);

SA[:]'*envB[:]

#########
basis = load("simulation/hb_g11_D2_X32/basis.jld2", "basis")
ts = load("simulation/hb_g11_D2_X32/basis.jld2", "ts");
C1, C2, C3, C4 = ts1.Cs
E1, E2, E3, E4 = ts1.Es
n_dm = iPEPS.get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4);
ndm_Ad = iPEPS.tcon([n_dm, ts.Ad], [[-1,-2,-3,-4,1,2,3,4], [1,2,3,4,-5]]);
ndm_A = iPEPS.tcon([n_dm, ts.A], [[1,2,3,4,-1,-2,-3,-4], [1,2,3,4,-5]])

[iPEPS.tcon([ndm_Ad, reshape(basis[:, i], size(ts.A))], [[1,2,3,4,5], [1,2,3,4,5]]) for i in axes(basis, 2)]

[iPEPS.tcon([conj(ndm_A), reshape(basis[:, i], size(ts.A))], [[1,2,3,4,5], [1,2,3,4,5]]) for i in axes(basis, 2)]

ndm_SA = iPEPS.tcon([n_dm, SA], [[1,2,3,4,-1,-2,-3,-4], [1,2,3,4,-5]])

bsa = basis'*ndm_SA[:] 
bsa[ bsa |> real .> 1e-10 ]

exci_n'*ndm_SA[:]

M = randn(2,2);
@ein B1[m1,m2, m3, m4,m5] := M[m2,p1] * ts.A[m1,p1,m3,m4,m5] 
@ein B2[m1,m2, m3, m4,m5] :=  M[p1,m4] * ts.A[m1,m2,m3,p1,m5]

B = B1 .- B2
ts1 = setproperties(ts, B=B, Bd=conj(B));
conv_fun(_x) = iPEPS.get_all_norm(_x)[1] 
# conv_fun(_x) = iPEPS.get_es_energy(_x, H) / iPEPS.get_all_norm(_x)[1]
ts1, _ = iPEPS.run_ctm(ts1, conv_fun = conv_fun);

nB = iPEPS.tcon([n_dm, B], [[-1,-2,-3,-4,1,2,3,4], [1,2,3,4,-5]]);

B[:]'*nB[:]