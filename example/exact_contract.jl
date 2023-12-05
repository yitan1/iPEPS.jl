using OMEinsum
using iPEPS, JLD2
using LinearAlgebra
using Zygote
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
    # Lb = [Lb1, Lb2, Lb3]

    # Ls0 = [La, La, La, La]

    res = zeros(ComplexF64, 4^12)


    res += tr_LL(mul_LL(mul_LL(Lb1, La), La), La);
    res += tr_LL(mul_LL(mul_LL(Lb2, La), La), La);
    res += tr_LL(mul_LL(mul_LL(Lb3, La), La), La);

    res += tr_LL(mul_LL(mul_LL(La, Lb1), La), La);
    res += tr_LL(mul_LL(mul_LL(La, Lb2), La), La);
    res += tr_LL(mul_LL(mul_LL(La, Lb3), La), La);

    res += tr_LL(mul_LL(mul_LL(La, La), Lb1), La);
    res += tr_LL(mul_LL(mul_LL(La, La), Lb2), La);
    res += tr_LL(mul_LL(mul_LL(La, La), Lb3), La);

    res += tr_LL(mul_LL(mul_LL(La, La), La), Lb1);
    res += tr_LL(mul_LL(mul_LL(La, La), La), Lb2);
    res += tr_LL(mul_LL(mul_LL(La, La), La), Lb3);


    res
end

function get_energy(A, B, h)
    phi1 = reshape(AB4x3(A, B), 4^4, 4, 4^2, 4, 4^4);
    phi1d = reshape(AB4x3(conj.(A), conj.(B)), 4^4, 4, 4^2, 4, 4^4);
    @ein dm[m1, m2,m3,m4] := phi1[p1, m1, p2, m2, p3] * phi1d[p1, m3, p2, m4, p3];
    dm = reshape(dm, 4^2, 4^2);
    n0 = tr(dm)
    eh = tr(dm*h)
    eh, n0
end

function get_EN(A, B, h)
    Bi = reshape(B, size(A))
    eh, nh = get_energy(A, Bi, h[1])

    Ah = permutedims(A, [4,1,2,3,5])
    Bih = permutedims(Bi, [4,1,2,3,5])
    ev, nv = get_energy(Ah, Bih, h[2])

    e = eh + ev
    n = (nh + nv)/2

    e, n
end

function run(A, Bi, h)
    (e, n), back = Zygote.pullback(x -> get_EN(A, x, h), Bi)
    
    gradH = back((1, nothing))[1];
    gradN = back((nothing, 1))[1];

    println("e: $e, n: $n")

    gradH/2, gradN/2
end

function main(A, h, n)
    bs = Matrix{ComplexF64}(I, 64, 64);
    if n > 1
        effH = load("exact_HN.jld2", effH)
        effN = load("exact_HN.jld2", effN)
    else
        effH = zeros(ComplexF64, 64, 64)
        effN = zeros(ComplexF64, 64, 64)
    end
    for i = n:64
        Bi = bs[:, i]
        (e, n), back = Zygote.pullback(x -> get_EN(A, x, h), Bi);
        gradH = back((1, nothing))[1];
        gradN = back((nothing, 1))[1];

        effH[:, i] = bs'*gradH/2
        effN[:, i] = bs'*gradN/2
        jldsave("exact_HN.jld2", effH = effH, effN = effN)
        println("finish $i/64")
        GC.gc()
    end

    jldsave("exact_HN.jld2", effH = effH, effN = effN)
end

h = honeycomb(1, 1, dir = "XX");
A = init_hb_gs(2, dir = "XX");
e, n = get_EN(A, A, h)

h1 = h[1] - real(e/n/2)*Matrix{ComplexF64}(I, 16, 16)
h2 = h[2] - real(e/n/2)*Matrix{ComplexF64}(I, 16, 16)
hr = [h1, h2]

main(A, hr, 1)

eh = get_energy(A, A, h[1])
Ah = permutedims(A, [4,1,2,3,5])
ev = get_energy(Ah, Ah, h[2])

f1 = load("exact_HN_wp.jld2")
effH = f1["effH"]
effN = f1["effN"]

H = (effH + effH')/2
N = (effN + effN')/2

ev_N, P = eigen(N)
idx = sortperm(real.(ev_N))[end:-1:1]
ev_N = ev_N[idx]
selected = (ev_N/maximum(ev_N) ) .> 1e-3
P = P[:,idx]
P = P[:,selected]
N2 = P' * N * P
H2 = P' * H * P
H2 = (H2 + H2') /2 
N2 = (N2 + N2') /2
es0, vecs = eigen(H2,N2)

bs = Matrix{ComplexF64}(I, 64, 64);
exci_n = bs*P*vecs;

using CairoMakie

s1 = iPEPS.sigmax
s2 = iPEPS.sigmax
op1 = iPEPS.tout(s1, s2)
SA = iPEPS.tcon([A, op1], [[-1,-2,-3,-4,1], [-5,1]]);

# gh, gn = run(A, SA, hr)
# jldsave("gn_x", gn = gn)
gn = load("gn_SA.jld2", "gn")
# gn = load("gn_x", "gn")
sk = exci_n'*gn[:]

y1 = abs.(sk)./maximum(abs.(sk))
scatter( es0*12, y1)

ei, ni = get_EN(A, exci_n[:,1], hr);

Bi = exci_n[:,8]
Bi = reshape(B, 2, 2, 2, 2, 4);
# Bi = iPEPS.get_vison(2)
A = ts.A
La = get_ABC(A,A,A);
Lb2 = get_ABC(A,Bi,A);
psi1 = tr_LL(mul_LL(mul_LL(La, Lb2), La), La);

Lb1 = get_ABC(Bi, A,A);
psi1 = tr_LL(mul_LL(mul_LL(La, La), Lb1), La);

psi1 = tr_LL(mul_LL(mul_LL(La, Lb1), La), La);
psi1 = tr_LL(mul_LL(mul_LL(La, La), Lb2), La);

function m_wp(psi1)
    psi1 = reshape(psi1, 4^3, 4, 4, 4, 4, 4, 4^4)
    psi1d = conj(psi1)
    ndm = tcon([psi1, psi1d], [[1,-1,-2,2,-3,-4,3], [1,-5,-6,2,-7,-8,3]])


    w1 = iPEPS.tout(iPEPS.sI, iPEPS.sigmax)
    w2 = iPEPS.tout(iPEPS.sigmaz, iPEPS.sigmay)
    w3 = iPEPS.tout(iPEPS.sigmay, iPEPS.sigmaz)
    w4 = iPEPS.tout(iPEPS.sigmax, iPEPS.sI)

    @ein nwp[m1, m2, m3, m4, m5, m6, m7, m8] := ndm[p1, p2, p3, p4, m5, m6, m7, m8,] * w1[m1, p1] * w2[m2, p2] * w3[m3, p3] * w4[m4, p4];

    nwp = reshape(nwp, 4^4, 4^4)
    wp = tr(nwp)
    ndm = reshape(ndm, 4^4, 4^4)
    n = tr(ndm)

    wp, n
end
wp, n = m_wp(psi1);

psi1'*psia

psia = tr_LL(mul_LL(mul_LL(La, La), La), La);
wp, n = m_wp(psia);


Lbs = get_ABC(A,SA,A);
psisa = tr_LL(mul_LL(mul_LL(La, Lbs), La), La);


# s1 = iPEPS.sigmax
# s2 = iPEPS.sigmax
# op1 = iPEPS.tout(s1, s2)
# A = ts.A
# SA = iPEPS.tcon([A, op1], [[-1,-2,-3,-4,1], [-5,1]]);

# wka = conj(SA[:]'*envB*P*vecs)[:]

# y = zeros(ComplexF64, size(exci_n, 2))

# n0 = transpose(AB4x3(conj(A), conj(A)))*AB4x3(A, A)

# B = reshape(exci_n[:, i], size(A));

# # phi2 = transpose(AB4x3(conj(A), conj(A)))


# f = Figure(xlabelfont = 34, ylabelfont = 34)
# ax = Axis(f[1, 1])
# # scatter!(ax, es, abs.(y))
# scatter!(ax, es, abs.(wka))
# # axislegend()
# # xlims!(ax, low = -1, high = 6)
# f
