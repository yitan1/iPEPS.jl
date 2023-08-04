# H1 = -XX - h*Z1/2 - h*Z2/2
# H = -SXSX*4 - h*Z1/2 * 2 - h*Z2/2 *2
function ising1(h = 3)
    xx = tcon([Sx, Sx], [[-1,-2], [-3, -4]]) * 2
    z1 = tcon([Sz, SI], [[-1,-2], [-3, -4]]) / 2
    z2 = tcon([Sz, SI], [[-1,-2], [-3, -4]]) / 2

    H = -xx - z1 - z2

    H
end

function get_identity(d = 2)
    H = tcon([SI, SI], [[-1,-2], [-3, -4]]) 

    H
end

function get_ots(ts, pepoN)
    pepoC, pepoE, pepoA = pepoN

    A = ts.A
    Ad = ts.Ad .|> Float64
    newA = tcon([A, pepoA], [[-1,-3,-5,-7,1],[-2,-4,-6,-8,1,-9]])
    dim = size(newA)
    newA=reshape(newA, dim[1]*dim[2], dim[3]*dim[4], dim[5]*dim[6], dim[7]*dim[8], dim[9])

    @ein newC1[m1,m2,m3,m4,m5,m6] := A[p3,p4,m1,m4, p1]*pepoC[1][m2,m5,p1,p2]*Ad[p3,p4,m3,m6,p2] 
    dim=size(newC1)
    newC1=reshape(newC1,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6])

    @ein newC2[m1,m2,m3,m4,m5,m6] := A[p3,m1,m4, p4, p1]*pepoC[2][m2,m5,p1,p2]*Ad[p3,m3,m6,p4,p2] 
    dim=size(newC2)
    newC2=reshape(newC1,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6])

    @ein newC3[m1,m2,m3,m4,m5,m6] := A[m1,m4,p3,p4, p1]*pepoC[3][m2,m5,p1,p2]*Ad[m3,m6,p3,p4,p2] 
    dim=size(newC3)
    newC3=reshape(newC3,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6])

    @ein newC4[m1,m2,m3,m4,m5,m6] := A[m1,p3,p4,m4, p1]*pepoC[4][m2,m5,p1,p2]*Ad[m3,p3,p4,m6,p2]
    dim=size(newC4)
    newC4=reshape(newC4,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6])

    @ein newE1[m1,m2,m3,m4,m5,m6,m7,m8,m9] := A[p3,m1,m7,m4,p1]*pepoE[1][m2,m5,m8,p1,p2]*Ad[p3,m3,m9,m6,p2] 
    dim=size(newE1)
    newE1=reshape(newE1,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6], dim[7]*dim[8], dim[9])

    @ein newE2[m1,m2,m3,m4,m5,m6,m7,m8,m9] := A[m1,m7,m4,p3,p1]*pepoE[2][m2,m5,m8,p1,p2]*Ad[m3,m9,m6,p3,p2] 
    dim=size(newE2)
    newE2=reshape(newE2,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6], dim[7]*dim[8], dim[9])

    @ein newE3[m1,m2,m3,m4,m5,m6,m7,m8,m9] := A[m7,m1,p3,m4,p1]*pepoE[3][m2,m5,m8,p1,p2]*Ad[m9,m3,p3,m6,p2] 
    dim=size(newE3)
    newE3=reshape(newE3,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6], dim[7]*dim[8], dim[9])

    @ein newE4[m1,m2,m3,m4,m5,m6,m7,m8,m9] := A[m1,p3,m4,m7, p1]*pepoE[4][m2,m5,m8,p1,p2]*Ad[m3,p3,m6,m9,p2] 
    dim=size(newE4)
    newE4=reshape(newE4,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6], dim[7]*dim[8], dim[9])

    newCs = [newC1, newC2, newC3, newC4]
    newEs = [newE1, newE2, newE3, newE4]

    ts = setproperties(ts, A = newA, Ad = Ad, Cs = newCs, Es = newEs)

    ts
end

function init_pepo(H0, tau)
    AL,Ah,AR= h2pepo(H0, tau)
    AU,Av,AD= h2pepo(H0, tau)

    A = tcon([Ah, Av], [[-2, -4, -5, 1], [-1, -3, 1, -6]])

    C1 = tcon([AL, AU], [[-3, 1, -2], [1, -4, -1]])
    C2 = tcon([AR, AU], [[-1, -3, 1], [1, -4, -2]])
    C3 = tcon([AR, AD], [[-2, 1, -4], [-1, -3, 1]])
    C4 = tcon([AL, AD], [[1, -4, -2], [-1, -3, 1]])

    T1 = tcon([Ah, AU], [[-1, -2, -4, 1], [1, -5, -3]])
    T2 = tcon([Av, AR], [[-1, -2, 1, -5], [-3, -4, 1]])
    T3 = tcon([Ah, AD], [[-1, -2, 1, -5], [-3, -4, 1]])
    T4 = tcon([Av, AL], [[-1, -2, 1, -5], [-4, 1, -3]])

    Cs = [C1,C2,C3,C4]
    Es = [T1,T2,T3,T4] 
    pepoN = [Cs, Es, A]

    pepoN
end

function h2pepo(H0, tau)
    d = size(H0,1)
    H = tau*reshape(H0, d*d, d*d) |> exp
    U, S, Vt = svd(H)

    V = Vt'
    S1 = sqrt.(S)
    
    L = U*diagm(S1)
    R = diagm(S1)*V
    # @show L*R - H

    # L:
    #  1 --\
    #       --- 3
    #  2 --/
    # R:
    #      /--- 2
    #  1---
    #      \--- 3
    L = reshape(L, d, d, :)
    R = reshape(R, :, d, d)

    A = tcon([L, R], [[-3, 1, -2], [-1,1,-4]])

    # pepo = tcon([hp, vp], [[-2, -4, -5, 1], [-1, -2, 1, -6]])
    
    L, A, R
end