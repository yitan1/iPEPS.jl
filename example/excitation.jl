using PhyOperators
using iPEPS
using MKL
using LinearAlgebra

using JLD2



A = load("example/gs_ising_D2_chi30.jld2")["A"]
phi = iPEPS.IPEPS(A);

# h = rotate_heisenberg()
h = spinmodel((-1,0,0), (0,0,-3))
ht = reshape(h, 2, 2, 2, 2)
# hr = h - iPEPS.get_energy(iPEPS.symmetrize(A), iPEPS.symmetrize(A), ht, ht; chi = 30)
tval, tvec = optimize_ES(1pi, 0.0, phi, ht, ht; chi = 30)

#################
env = iPEPS.get_envtensor(phi; chi = 30) ;
nrm0 = iPEPS.get_norm(env)
phi = iPEPS.IPEPS(iPEPS.get_A(phi) / sqrt(abs(nrm0)) )
Bn = iPEPS.get_tangent_basis(phi; chi = 30)
E0 = iPEPS.get_energy(iPEPS.get_A(phi), ht, ht)
id = Matrix{Float64}(I, size(ht,1)*size(ht,2), size(ht,3)*size(ht,4))
h_hor = ht .- E0*reshape(id, size(ht))
h_ver = ht .- E0*reshape(id, size(ht))

H, N = iPEPS.eff_hamitonian_norm(h_hor, h_ver, .0, .0, phi, Bn; chi = 30)
H = (H + H')/2
N = (N + N')/2
ev_N, P = eigen(N)
idx = sortperm(real.(ev_N))[end:-1:1]
ev_N = ev_N[idx]
selected = (ev_N/maximum(ev_N) ) .> 1e-3
P = P[:,idx]
P = P[:,selected]
N2 = P' * N * P
H2 = P' * H * P
eigen(H2,N2)
########################

kx1 = ones(5)*pi
kx2 = [i for i= 9:-1:0] *pi/10
kx3 = [i for i=1:5]*pi/5
ky1 = [i for i=1:5]*pi/5
ky2 = [i for i= 9:-1:0] *pi/10
ky3 = ones(5)*pi

kx = vcat(kx1,kx2,kx3)
ky = vcat(ky1, ky2, ky3)

Es = zeros(ComplexF64, length(kx), 10)
for i in eachindex(kx)
    val, vec = optimize_ES(kx[i], ky[i], phi, ht, ht; chi = 30)
    if length(val) >= 10
        Es[i,:] = val[1:10]
    else
        Es[i,1:length(val)] = val[1:length(val)]
    end
    @show i, Es[i, 1]
end

jldsave("example/es2_ising_D2_chi30_A.jld2"; kx, ky, Es)

Es = load("example/es_ising_D2_chi30_A.jld2")["Es"] 
Es2 = load("example/es2_ising_D2_chi30_A.jld2")["Es"] 
Es - Es2 |> sum

let
using Plots
Es = load("example/es2_ising_D2_chi30_A.jld2")["Es"] 
# Es = Es.+ 0.66*4
x1 = collect(1:5)
x2 = collect(5.5:0.5:10)
x3 = collect(11:15)
x = vcat(x1,x2,x3)
p = scatter(x, real.(Es[:, 1]);label = false)

for i= 1:5
    plot!(p, x, real.(Es[:,i]);label = false)
end
p

end