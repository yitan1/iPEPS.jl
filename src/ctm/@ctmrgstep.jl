function ctmrgstep(C10,C20,C30,C40,T10,T20,T30,T40,A,χ)
    chi,D=size(C10)[1],size(A)[1];

    # generate C
    CTT=ein"(ef,aeg),fbh->abgh"(C10,T40,T10); C1 =ein"abgh,hcdg->adbc"(CTT,A)
    CTT=ein"(ef,aeg),fbh->abgh"(C20,T10,T20); C2 =ein"abgh,ghcd->adbc"(CTT,A)
    CTT=ein"(ef,aeg),fbh->abgh"(C30,T20,T30); C3 =ein"abgh,dghc->adbc"(CTT,A)    
    CTT=ein"(ef,aeg),fbh->abgh"(C40,T30,T40); C4 =ein"abgh,cdgh->adbc"(CTT,A)
    C1,C2,C3,C4=map(x->reshape(x,chi*D,chi*D),(C1,C2,C3,C4))    

    # trucation @(p1,p5) and projection without trucation @(p2,p3,p4)
    p1,S,p5=wrap_svd(C1*C2*C3*C4,χ)        
    vals=S/S[1]
    C1=p1'*C1; _,_,p2=svd1(C1);
    C2=p2'*C2; _,_,p3=svd1(C2)
    C3=p3'*C3; _,_,p4=svd1(C3)

    # RG after projection
    C1,C2,C3,C4=C1*p2,C2*p3,C3*p4,p4'*C4*p5

    p1,p2,p3,p4,p5=map(x->reshape(x,chi,D,:),(p1,p2,p3,p4,p5))    
    PTA=ein"(dha,def),fgch->aegc"(conj(p2),T10,A); T1=ein"aegc,egb->abc"(PTA,p2)
    PTA=ein"(dha,def),hfgc->aegc"(conj(p3),T20,A); T2=ein"aegc,egb->abc"(PTA,p3)
    PTA=ein"(dha,def),chfg->aegc"(conj(p4),T30,A); T3=ein"aegc,egb->abc"(PTA,p4)
    PTA=ein"(dha,def),gchf->aegc"(conj(p5),T40,A); T4=ein"aegc,egb->abc"(PTA,p1)

    return C1,C2,C3,C4,T1,T2,T3,T4,vals
end

function rg_step1(ts, χ)
    C1, C2, C3, C4 = ts.Cs
    T1, T2, T3, T4 = ts.Es
    A = tcon([ts.A, ts.Ad], [[-7,-1,-3,-5,1], [-8,-2,-4,-6, 1]])
    D = size(A,1)
    A = reshape(A, D*D, D*D, D*D, D*D)
    C1, C2, C3, C4, T1, T2, T3, T4, s = ctmrgstep(C1, C2, C3, C4, T1, T2, T3, T4, A, χ)

    C1 = renormalize(C1)
    C2 = renormalize(C2)
    C3 = renormalize(C3)
    C4 = renormalize(C4)

    T1 = renormalize(T1)
    T2 = renormalize(T2)
    T3 = renormalize(T3)
    T4 = renormalize(T4)

    ts = setproperties(ts, Cs=[C1, C2, C3, C4], Es=[T1, T2, T3, T4])

    ts, s
end

function get_gs_energy1(ts, H)
    ts = convert_order_back(ts)

    get_gs_energy(ts, H)
end

function convert_order_to(ts)
    C1, C2, C3, C4 = ts.Cs
    T1, T2, T3, T4 = ts.Es
    A = ts.A
    D = size(A,1)
    
    C10 = permutedims(C1, [2,1])
    C20 = permutedims(C4, [2,1])
    C30 = permutedims(C3, [2,1])
    C40 = C2

    T10 = reshape(T4, size(T4, 1), size(T4, 2), D*D)

    T20 = reshape(T3, size(T3, 1), size(T3, 2), D*D)

    T30 = reshape(T2, size(T2, 1), size(T2, 2), D*D)
    T30 = permutedims(T30, [2,1,3])

    T40 = reshape(T1, size(T1, 1), size(T1, 2), D*D)
    T40 = permutedims(T40, [2,1,3])

    ts = setproperties(ts, Cs=[C10, C20, C30, C40], Es=[T10, T20, T30, T40])

    ts
end
function convert_order_back(ts)
    C1, C2, C3, C4 = ts.Cs
    T1, T2, T3, T4 = ts.Es
    A = ts.A

    D = size(A,1)

    C10 = permutedims(C1, [2,1])
    C30 = permutedims(C3, [2,1])
    C20 = permutedims(C4, [2,1])
    C40 = C2
    
    T10 = begin
        T10 = permutedims(T4, [2,1,3])
        reshape(T10, size(T10, 1), size(T10, 2), D, D)
    end

    T20 = begin
        T20 = permutedims(T3, [2,1,3])
        reshape(T20, size(T20, 1), size(T20, 2), D, D)
    end

    T30 = reshape(T2, size(T2, 1), size(T2, 2), D, D)

    T40 = reshape(T1, size(T1, 1), size(T1, 2), D, D)

    ts = setproperties(ts, Cs=[C10, C20, C30, C40], Es=[T10, T20, T30, T40])

    ts
end

