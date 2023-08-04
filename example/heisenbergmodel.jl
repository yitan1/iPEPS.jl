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

####
H = heisenberg(1);

rng = MersenneTwister(3);

A = randn(rng, Float64, 2,2,2,2,2);

res = optim_gs(H, A, "")

prepare_basis(H, "")

optim_es(0., 0., "")

op = iPEPS.Sz
es, swk0 = compute_spectral(op, .0, .0, "")

x, y = plot_spectral(es, swk0)

scatter(es, swk0)


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

