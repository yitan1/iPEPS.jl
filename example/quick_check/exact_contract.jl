using OMEinsum
using iPEPS, JLD2
using LinearAlgebra
function get_ABC(A, B, C)
    D = size(A,1)
    d = size(A, 5)
    @ein AB[m1, m2, m3, m4, m5, m6, m7, m8] := A[m1, m2, p1, m5, m7] * B[p1, m3, m4, m6, m8]

    @ein ABC[m1, m2, m3, m4, m5, m6, m7, m8, m9] := AB[p1, m1, m2, p2, m4, m5, m7, m8] * C[p2, m3, p1, m6, m9]

    ABC = reshape(ABC, D^3, D^3, d^3)
    
    ABC
end

function mul_LL(L1, L2)
    @ein LL[m1, m2, m3, m4] := L1[m1, p1, m3] * L2[p1, m2, m4]
    LL = reshape(LL, size(LL,1), size(LL,2), size(LL,3)*size(LL,4))

    LL
end

function tr_LL(L1, L2)
    @ein LL[m3, m4] := L1[p2, p1, m3] * L2[p1, p2, m4]
    LL = reshape(LL, :)
    LL
end

function AB4x3(A, B)
    n0 = norm(A)
    La = get_ABC(A,A,A)./n0;
    Lb1 = get_ABC(B,A,A)./n0;
    Lb2 = get_ABC(A,B,A)./n0;
    Lb3 = get_ABC(A,A,B)./n0;
    Lb = [Lb1, Lb2, Lb3]

    Ls = [La, La, La, La]

    res = zeros(ComplexF64, 4^12)

    for i = 1:4
        for j = 1:3
            Ls[i] = Lb[j] 
            res += tr_LL(mul_LL(mul_LL(Ls[1], Ls[2]), Ls[3]), Ls[4]);
        end
    end
    res
end

B = Matrix{ComplexF64}(I, 64, 64)

ts = load("simulation/hb_g11_D2_X32/basis.jld2")["ts"];
basis = load("simulation/hb_g11_D2_X32/basis.jld2", "basis")
es, vecs, P, envB = compute_es(0, 0, ""; disp = true);
exci_n = basis*P*vecs;

s1 = iPEPS.sigmax
s2 = iPEPS.sigmax
op1 = iPEPS.tout(s1, s2)
A = ts.A
SA = iPEPS.tcon([A, op1], [[-1,-2,-3,-4,1], [-5,1]]);

wka = conj(SA[:]'*envB*P*vecs)[:]

y = zeros(ComplexF64, size(exci_n, 2))

n0 = transpose(AB4x3(conj(A), conj(A)))*AB4x3(A, A)
for i in axes(exci_n, 2)
    # i = 1
    B = reshape(exci_n[:, i], size(A));
    # phi1 = AB4x3(A, A)
    # phi2 = transpose(AB4x3(conj(A), conj(A)))
    y[i] = transpose(AB4x3(conj(A), conj(B)))*AB4x3(A, SA)./n0
    println(i)
end

f = Figure(xlabelfont = 34, ylabelfont = 34)
ax = Axis(f[1, 1])
# scatter!(ax, es, abs.(y))
scatter!(ax, es, abs.(wka))
# axislegend()
# xlims!(ax, low = -1, high = 6)
f

i = 1
B = reshape(exci_n[:, i], size(A));
# phi1 = AB4x3(A, A)
# phi2 = transpose(AB4x3(conj(A), conj(A)))
transpose(AB4x3(conj(A), conj(B)))*AB4x3(A, B)./n0
