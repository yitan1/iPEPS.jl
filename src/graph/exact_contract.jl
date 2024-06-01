
###############################################################################

function run_wp_exact(ts, B)
    Bi = reshape(B, size(ts.A));
    A = ts.A
    # A = init_hb_gs(2, dir = "XX")
    La = get_ABC(A,A,A);
    Lb1 = get_ABC(Bi, A,A);
    Lb2 = get_ABC(A,Bi,A);

    psi1 = tr_LL(mul_LL(mul_LL(La, Lb1), La), La);
    psi2 = tr_LL(mul_LL(mul_LL(La, Lb2), La), La);
    psi3 = tr_LL(mul_LL(mul_LL(La, La), Lb1), La);
    psi4 = tr_LL(mul_LL(mul_LL(La, La), Lb2), La);

    wp1, n1 = m_wp(psi1)
    wp2, n2 = m_wp(psi2)
    wp3, n3 = m_wp(psi3)
    wp4, n4 = m_wp(psi4)

    wp = wp1 + wp2 + wp3 + wp4
    wp = wp2 + wp3
    nB = (n2 + n3)/2
    nB = (n1 + n2 + n3 + n4)/4

    fprint("w1: $(wp1/nB)    n1: $(n1)\nw2: $(wp2/nB)    n2: $(n2)\nw3: $(wp3/nB)    n3: $(n3)\nw4: $(wp4/nB)    n4: $(n4)")
    # fprint("w2: $(wp2/nB)    n2: $(n2)\nw3: $(wp3/nB)    n3: $(n3)")

    wp, nB
end

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