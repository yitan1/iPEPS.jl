function compute_gs_energy(A::AbstractArray, H)
    ts0 = CTMTensors(A)
    println("\n ---- Start to find fixed points -----")
    # conv_fun(_x) = get_gs_energy(_x, H)
    conv_fun = nothing
    ts, _ = run_ctm(ts0; conv_fun = conv_fun)
    println("---- End to find fixed points ----- \n")

    # ts = normalize_gs(ts0)
    nrm0 = get_gs_norm(ts::CTMTensors)
    @show nrm0

    get_gs_energy(ts, H), nrm0
end

function get_gs_energy(ts::CTMTensors, H)
    E, _ = get_E_N(ts, H)
    E[1] + E[2]
end

function get_es_energy(ts::CTMTensors, H)
    roh, rov = get_dms(ts, only_gs = false)
    Nh = wrap_tr(roh)  |> real
    Nv = wrap_tr(rov) |> real
    roh = roh / Nh
    rov = rov / Nv
    Eh = wrap_tr(H[1]*roh) |> real
    Ev = wrap_tr(H[2]*rov) |> real
    # @show rov
    E = Eh + Ev
    N = Nh + Nv
    # @show E[1], E[4]

    E[4]
end

function get_E_N(ts::CTMTensors, H)
    roh, rov = get_dms(ts)
    Nh = tr(roh)
    Nv = tr(rov)
    roh = roh ./ Nh
    rov = rov ./ Nv
    Eh = tr(H[1]*roh)
    Ev = tr(H[2]*rov)
    E = [Eh, Ev]
    N = [Nh, Nv]

    E, N
end

function get_gs_norm(ts::CTMTensors)
    A = ts.A
    Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)

    ndm_Ad = tcon([n_dm, Ad], [[-1,-2,-3,-4,1,2,3,4], [1,2,3,4,-5]])
    nrm0 = tcon([ndm_Ad, A], [[1,2,3,4,5], [1,2,3,4,5]])

    nrm0[1]
end

function get_all_norm(ts::CTMTensors)
    A    = get_all_A(ts)
    Ad   = get_all_Ad(ts)
    C1, C2, C3, C4 = get_all_Cs(ts)
    E1, E2, E3, E4 = get_all_Es(ts)
    n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)

    ndm_A = tcon([n_dm, A], [[-1,-2,-3,-4,1,2,3,4], [1,2,3,4,-5]])
    all_norm = tcon([ndm_A, Ad], [[1,2,3,4,5], [1,2,3,4,5]])

    all_norm, ndm_A[2]
end

function get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)
    C1E1 = tcon([C1, E1], [[-1,1], [1,-2,-3,-4]])
    CEC2 = tcon([C1E1, C2], [[-1,1,-3,-4], [1,-2]])
    CECE4 = tcon([CEC2, E4], [[1,-2,-3,-5], [1,-1,-4,-6]])

    C4E3 = tcon([C4, E3], [[-1, 1], [1, -2, -3, -4]])
    CEC3 = tcon([C4E3, C3], [[-1,1,-3,-4], [-2,1]])
    CECE2 = tcon([CEC3, E2], [[-1,1,-3,-5], [-2,1,-4,-6]])

    n_dm = tcon([CECE4, CECE2], [[1,2, -1,-2,-5,-6], [1,2, -3, -4, -7, -8]])

    n_dm
end

function get_dms(ts::CTMTensors; only_gs = true)
    if only_gs
        A    = ts.A
        Ad   = ts.Ad
        C1   = ts.Cs[1]
        C2   = ts.Cs[2]
        C3   = ts.Cs[3]
        C4   = ts.Cs[4]
        E1   = ts.Es[1]
        E2   = ts.Es[2]
        E3   = ts.Es[3]
        E4   = ts.Es[4]

        ts_h = [C1, C2, C3, C4, E1, E1, E2, E3, E3, E4, A, A, Ad, Ad]
        ts_v = [C1, C2, C3, C4, E1, E2, E2, E3, E4, E4, A, A, Ad, Ad]
    else
        A    = get_all_A(ts)
        Ad   = get_all_Ad(ts)
        C1, C2, C3, C4 = get_all_Cs(ts)
        E1, E2, E3, E4 = get_all_Es(ts)
        px = get(ts.Params, "px", .0)
        py = get(ts.Params, "py", .0)

        ts_h = [C1, shift(C2, px), shift(C3, px), C4, E1, shift(E1,px), shift(E2, px), E3, shift(E3, px), E4, A, shift(A, px), Ad, shift(Ad, px)]

        ts_v = [C1, C2, shift(C3, py), shift(C4, py), E1, E2, shift(E2, py), shift(E3, py), E4, shift(E4, py), A, shift(A, py), Ad, shift(Ad, py)]
    end

    roh = get_dm_h(ts_h...)
    rov = get_dm_v(ts_v...)

    return roh, rov
end
"""
    get_dm_hor

return density matrix(d,d,d,d) of following diagram
```
C1 -- E1l -- E1r -- C2
|     ||     ||      |
E4 == AAl< =>AAr == E2  = ρₕ
|     ||     ||      |  
C4 -- E3l -- E3r -- C3
```
"""
function get_dm_h(C1, C2, C3, C4, E1l, E1r, E2, E3l, E3r, E4, Al, Ar, Adl, Adr)
    #left
    LU = begin
        C1E1 = tcon([C1, E1l], [[-1,1], [1, -2, -3, -4]])
        tcon([C1E1, Adl], [[-1,-2, -3, 1], [1, -4,-5,-6,-7]])
    end
    LB = begin
        C4E3 = tcon([C4, E3l], [[-1,1], [1,-2, -3, -4]])    # [t, r], [l, r, bra, ket] -> [t, r, bra, ket]
        CEE4 = tcon([E4, C4E3], [[-1,1, -3,-5], [1, -2, -4, -6]])  # [t, b, bra, ket], [t, r, bra, ket] -> 
        tcon([CEE4, Al], [[-1,-2,1,2,-4,-5], [-3, 1, 2, -6, -7]])  
    end  
    L = tcon([LB,LU], [[1,-2,2,3,4,-3,-5], [1,-1,2,3,4,-4,-6]])

    #right
    RU = begin
        C2E1 = tcon([C2, E1r], [[1,-2],[-1, 1, -3, -4]])
        tcon([C2E1, Ar], [[-1, -2, 1, -3], [1, -4, -5, -6, -7]])
    end 
    RB = begin
        C3E3 = tcon([C3, E3r], [[-1,1],[-2,1, -3,-4]])
        CEE2 = tcon([C3E3, E2], [[1,-2, -3, -5],[-1,1, -4,-6]])
        tcon([CEE2, Adr], [[-1,-2,-3,-4,1,2], [-5,-6,1,2,-7]])
    end
    R = tcon([RU,RB], [[-1,1,2,-3,3,4,-5],[1,-2,3,4,2,-4,-6]])

    roh = tcon([L,R],[[1,2,3,4,-1,-3],[1,2,3,4,-2,-4]])

    d = size(Al, 5)
    roh = wrap_reshape(roh, d*d, d*d)

    return roh
end

"""
    get_dm_v

return density matrix(d,d,d,d) of following diagram
```
C1 --  E1 -- C2
|      |     |   
E4u -- Au-- E2u
|      Λ     |  =  ρᵥ
|      V     |
E4d -- Ad-- E2d  
|      |     |  
C4 --  E3 -- C3
```
"""
function get_dm_v(C1, C2, C3, C4, E1, E2u, E2d, E3, E4u, E4d, Au, Ad, Adu, Add)
    #up
    UL = begin
        C1E1 = tcon([C1, E1], [[-1,1], [1,-2,-3,-4]])
        CEE4 = tcon([C1E1, E4u], [[1,-2,-3,-5], [1,-1,-4,-6]])
        tcon([CEE4, Au], [[-1,-2,1,2,-3,-4], [1,2,-5,-6,-7]])
    end
    UR = begin
        C2E2 = tcon([C2,E2u], [[-1,1], [1,-2,-3,-4]])
        tcon([C2E2, Adu], [[-1,-2,-3,1],[-4,-5,-6,1,-7]])
    end
    U = tcon([UL,UR], [[-1,1,3,4,-3,2,-5], [1,-2,2,3,4,-4,-6]])

    #bottom
    BL = begin
        C4E3 = tcon([C4,E3], [[-1,1], [1,-2,-3,-4]])
        CEE4 = tcon([C4E3, E4d], [[1,-2,-4,-6],[-1,1,-3,-5]])
        tcon([CEE4, Add], [[-1,-2,-3,-4,1,2], [-5,1,2,-6,-7]])
    end
    BR = begin
        C3E2 = tcon([C3, E2d], [[1,-2], [-1,1,-3,-4]])
        tcon([C3E2, Ad], [[-1,-2,1,-3], [-4,-5,-6,1,-7]])
    end
    B = tcon([BL, BR], [[-1,1,2,3,-4,4,-6], [-2,1,4,-3,2,3,-5]])

    rov = tcon([U,B], [[1,2,3,4,-1,-3], [1,2,3,4,-2,-4]])

    d = size(Au, 5)
    rov = wrap_reshape(rov, d*d, d*d)

    return rov
end