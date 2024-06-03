function compute_gs_energy(A::AbstractArray, H, cfg)
    ts = CTMTensors(A, cfg)
    fprint("\n ---- Start to find fixed points -----")
    conv_fun(_x) = get_gs_energy(_x, H)[1]
    ts, _ = run_ctm(ts; conv_fun=conv_fun)
    fprint("---- End to find fixed points ----- \n")

    # ts = normalize_gs(ts0)
    nrm0 = get_gs_norm(ts::CTMTensors)
    @show nrm0

    get_gs_energy(ts, H)
end

function get_gs_energy(ts::CTMTensors, H)
    roh, rov = get_dms(ts)
    Nh = tr(roh)
    Nv = tr(rov)
    Na = (Nh + Nv) / 2

    roh = roh ./ Nh
    rov = rov ./ Nv
    Eh = tr(H[1] * roh)
    Ev = tr(H[2] * rov)

    E = Eh + Ev

    E, Na
end

function get_es_energy(ts::CTMTensors, H)
    roh, rov = get_dms(ts, only_gs=false)
    Nh = wrap_tr(roh[1]) |> real
    Nv = wrap_tr(rov[1]) |> real

    roh = roh * (1 / Nh)
    rov = rov * (1 / Nv)
    Eh = wrap_tr(H[1] * roh)
    Ev = wrap_tr(H[2] * rov)

    E = Eh + Ev
    # @show E[1], E[4]

    real(E[4])
end

function get_gs_norm(ts::CTMTensors)
    A = ts.A
    Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)

    ndm_Ad = tcon([n_dm, Ad], [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5]])
    nrm0 = tcon([ndm_Ad, A], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

    nrm0[1]
end

function get_all_norm(ts::CTMTensors)
    A = get_all_A(ts)
    Ad = get_all_Ad(ts)
    C1, C2, C3, C4 = get_all_Cs(ts)
    E1, E2, E3, E4 = get_all_Es(ts)
    n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)

    ndm_A = tcon([n_dm, A], [[1, 2, 3, 4, -1, -2, -3, -4], [1, 2, 3, 4, -5]])
    all_norm = tcon([ndm_A, Ad], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

    envB = ndm_A[2] / all_norm[1][1]

    Nb = tcon([ndm_A[2], ts.Bd], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    Nb = Nb / all_norm[1][1]

    Nb[1], envB
end

function get_all_norm1(ts::CTMTensors)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es
    B_C1, B_C2, B_C3, B_C4 = ts.B_Cs
    B_E1, B_E2, B_E3, B_E4 = ts.B_Es
    Bd_C1, Bd_C2, Bd_C3, Bd_C4 = ts.Bd_Cs
    Bd_E1, Bd_E2, Bd_E3, Bd_E4 = ts.Bd_Es
    n_tensors = [C1, C2, C3, C4, E1, E2, E3, E4]
    B_tensors = [B_C1, B_C2, B_C3, B_C4, B_E1, B_E2, B_E3, B_E4]
    # Bd_tensors = [Bd_C1, Bd_C2, Bd_C3, Bd_C4, Bd_E1, Bd_E2, Bd_E3, Bd_E4]

    n_dm = get_single_dm(n_tensors...)
    B_dm = zeros(size(n_dm))
    for i = 1:8
        cur_tensors = deepcopy(n_tensors)
        cur_tensors[i] = B_tensors[i]
        new_dm = get_single_dm(cur_tensors...)
        B_dm = B_dm + new_dm
    end

    ndm_Ad = tcon([n_dm, ts.Ad], [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5]])
    nrm0 = tcon([ndm_Ad, ts.A], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

    nrmB_open = (tcon([n_dm, ts.B], [[1, 2, 3, 4, -1, -2, -3, -4], [1, 2, 3, 4, -5]]) +
                 tcon([B_dm, ts.A], [[1, 2, 3, 4, -1, -2, -3, -4], [1, 2, 3, 4, -5]])) / nrm0[1]

    Nb = tcon([nrmB_open, ts.Bd], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

    Nb[1], nrmB_open
end

function get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)
    C1E1 = tcon([C1, E1], [[-1, 1], [1, -2, -3, -4]])
    CEC2 = tcon([C1E1, C2], [[-1, 1, -3, -4], [1, -2]])
    CECE4 = tcon([CEC2, E4], [[1, -2, -3, -5], [1, -1, -4, -6]])

    C4E3 = tcon([C4, E3], [[-1, 1], [1, -2, -3, -4]])
    CEC3 = tcon([C4E3, C3], [[-1, 1, -3, -4], [-2, 1]])
    CECE2 = tcon([CEC3, E2], [[-1, 1, -3, -5], [-2, 1, -4, -6]])

    n_dm = tcon([CECE4, CECE2], [[1, 2, -1, -2, -5, -6], [1, 2, -3, -4, -7, -8]])

    n_dm
end

function get_dms(ts::CTMTensors; only_gs=true)
    if only_gs
        A = ts.A
        Ad = ts.Ad
        C1 = ts.Cs[1]
        C2 = ts.Cs[2]
        C3 = ts.Cs[3]
        C4 = ts.Cs[4]
        E1 = ts.Es[1]
        E2 = ts.Es[2]
        E3 = ts.Es[3]
        E4 = ts.Es[4]

        ts_h = (C1, C2, C3, C4, E1, E1, E2, E3, E3, E4, A, A, Ad, Ad)
        ts_v = (C1, C2, C3, C4, E1, E2, E2, E3, E4, E4, A, A, Ad, Ad)
    else
        A = get_all_A(ts)
        Ad = get_all_Ad(ts)
        C1, C2, C3, C4 = get_all_Cs(ts)
        E1, E2, E3, E4 = get_all_Es(ts)
        px = get(ts.Params, "px", 0.0f0)
        py = get(ts.Params, "py", 0.0f0)

        ts_h = (C1, shift(C2, px), shift(C3, px), C4, E1, shift(E1, px), shift(E2, px), E3, shift(E3, px), E4, A, shift(A, px), Ad, shift(Ad, px))

        ts_v = (C1, C2, shift(C3, py), shift(C4, py), E1, E2, shift(E2, py), shift(E3, py), E4, shift(E4, py), A, shift(A, py), Ad, shift(Ad, py))
    end

    roh = get_dm_h(ts_h...)
    rov = get_dm_v(ts_v...)

    return roh, rov
end
"""
    get_dm_h

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
        C1E1 = tcon([C1, E1l], [[-1, 1], [1, -2, -3, -4]])
        tcon([C1E1, Adl], [[-1, -2, -3, 1], [1, -4, -5, -6, -7]])
    end
    LB = begin
        C4E3 = tcon([C4, E3l], [[-1, 1], [1, -2, -3, -4]])    # [t, r], [l, r, bra, ket] -> [t, r, bra, ket]
        CEE4 = tcon([E4, C4E3], [[-1, 1, -3, -5], [1, -2, -4, -6]])  # [t, b, bra, ket], [t, r, bra, ket] -> 
        tcon([CEE4, Al], [[-1, -2, 1, 2, -4, -5], [-3, 1, 2, -6, -7]])
    end
    L = tcon([LB, LU], [[1, -2, 2, 3, 4, -3, -5], [1, -1, 2, 3, 4, -4, -6]])

    #right
    RU = begin
        C2E1 = tcon([C2, E1r], [[1, -2], [-1, 1, -3, -4]])
        tcon([C2E1, Ar], [[-1, -2, 1, -3], [1, -4, -5, -6, -7]])
    end
    RB = begin
        C3E3 = tcon([C3, E3r], [[-1, 1], [-2, 1, -3, -4]])
        CEE2 = tcon([C3E3, E2], [[1, -2, -3, -5], [-1, 1, -4, -6]])
        tcon([CEE2, Adr], [[-1, -2, -3, -4, 1, 2], [-5, -6, 1, 2, -7]])
    end
    R = tcon([RU, RB], [[-1, 1, 2, -3, 3, 4, -5], [1, -2, 3, 4, 2, -4, -6]])

    roh = tcon([L, R], [[1, 2, 3, 4, -1, -3], [1, 2, 3, 4, -2, -4]])

    d = size(Al, 5)
    roh = wrap_reshape(roh, d * d, d * d)

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
        C1E1 = tcon([C1, E1], [[-1, 1], [1, -2, -3, -4]])
        CEE4 = tcon([C1E1, E4u], [[1, -2, -3, -5], [1, -1, -4, -6]])
        tcon([CEE4, Au], [[-1, -2, 1, 2, -3, -4], [1, 2, -5, -6, -7]])
    end
    UR = begin
        C2E2 = tcon([C2, E2u], [[-1, 1], [1, -2, -3, -4]])
        tcon([C2E2, Adu], [[-1, -2, -3, 1], [-4, -5, -6, 1, -7]])
    end
    U = tcon([UL, UR], [[-1, 1, 3, 4, -3, 2, -5], [1, -2, 2, 3, 4, -4, -6]])

    #bottom
    BL = begin
        C4E3 = tcon([C4, E3], [[-1, 1], [1, -2, -3, -4]])
        CEE4 = tcon([C4E3, E4d], [[1, -2, -4, -6], [-1, 1, -3, -5]])
        tcon([CEE4, Add], [[-1, -2, -3, -4, 1, 2], [-5, 1, 2, -6, -7]])
    end
    BR = begin
        C3E2 = tcon([C3, E2d], [[1, -2], [-1, 1, -3, -4]])
        tcon([C3E2, Ad], [[-1, -2, 1, -3], [-4, -5, -6, 1, -7]])
    end
    B = tcon([BL, BR], [[-1, 1, 2, 3, -4, 4, -6], [-2, 1, 4, -3, 2, 3, -5]])

    rov = tcon([U, B], [[1, 2, 3, 4, -1, -3], [1, 2, 3, 4, -2, -4]])

    d = size(Au, 5)
    rov = wrap_reshape(rov, d * d, d * d)

    return rov
end

"""
    get_dm4

return density matrix(d^4,d^4) of following diagram
```
C1 --  E1 -- E1 -- C2
|      | v   | v    |
E4 --  B1 -- B3 -- E2
|      | v   | v    |
E4 --  B2 -- B4 -- E2  
|      |     |      |
C4 --  E3 -- E3 -- C3
```
"""
function get_dm4(C1, C2, C3, C4, E1, E2, E3, E4, B1, B2, B3, B4)
    block1 = begin
        C1E1 = tcon([C1, E1], [[-1, 1], [1, -2, -3, -4]])
        CEE4 = tcon([C1E1, E4], [[1, -2, -3, -5], [1, -1, -4, -6]])
        CEEB = tcon([CEE4, B1], [[-1, -2, 1, 2, -3, -4], [1, 2, -5, -6, -7]])
        tcon([CEEB, conj(B1)], [[-1, -2, 1, 2, -3, -4, -7], [1, 2, -5, -6, -8]])
    end

    block2 = begin
        C4E3 = tcon([C4, E3], [[-1, 1], [1, -2, -3, -4]])
        CEE4 = tcon([C4E3, E4], [[1, -2, -4, -6], [-1, 1, -3, -5]])
        CEEB = tcon([CEE4, B2], [[-1, -2, 1, 2, -3, -4], [-5, 1, 2, -6, -7]])
        tcon([CEEB, conj(B2)], [[-1, -2, 1, 2, -3, -4, -7], [-5, 1, 2, -6, -8]])
    end
    block_L = tcon([block1, block2], [[1, -1, 2, -3, 3, -5, -7, -9], [1, -2, 2, -4, 3, -6, -8, -10]])

    block3 = begin
        C2E1 = tcon([C2, E1], [[1, -2], [-1, 1, -3, -4]])
        CEE2 = tcon([C2E1, E2], [[-1, 1, -3, -5], [1, -2, -4, -6]])
        CEEB = tcon([CEE2, B3], [[-1, -2, 1, 2, -3, -4], [1, -5, -6, 2, -7]])
        tcon([CEEB, conj(B3)], [[-1, -2, 1, 2, -3, -4, -7], [1, -5, -6, 2, -8]])
    end

    block4 = begin
        C3E3 = tcon([C3, E3], [[-1, 1], [-2, 1, -3, -4]])
        CEE2 = tcon([C3E3, E2], [[1, -2, -3, -5], [-1, 1, -4, -6]])
        CEEB = tcon([CEE2, B4], [[-1, -2, 1, 2, -3, -4], [-5, -6, 1, 2, -7]])
        tcon([CEEB, conj(B4)], [[-1, -2, 1, 2, -3, -4, -7], [-5, -6, 1, 2, -8]])
    end

    block_R = tcon([block3, block4], [[-1, 1, -3, 2, -5, 3, -7, -9], [1, -2, 2, -4, 3, -6, -8, -10]])

    dm4 = tcon([block_L, block_R], [[1, 2, 3, 4, 5, 6, -1, -2, -5, -6], [1, 2, 3, 4, 5, 6, -3, -4, -7, -8]])

    dm4
end

function run_4x4(ts, H)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = iPEPS.get_dm4(C1, C2, C3, C4, E1, E2, E3, E4, ts.A, ts.A, ts.A, ts.A)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:4]), :)

    nB = tr(n_dm)
    n_dm = n_dm ./ nB
    e0 = tr(H * n_dm)

    fprint("E0: $e0    nB: $(nB)")

    e0, nB
end
# TODO
function get_gs_energy_h2(ts::CTMTensors, H)
    s1, s2 = H

    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es
    A = ts.A
    A1, A2 = A, A
    A1d, A2d = ts.Ad, ts.Ad

    op_A1d, op_A2d = get_op_Ad2(s1, s2, A1d, A2d)

    eh, nh = energy_norm_3x4(C1, C2, C3, C4, E1, E1, E2, E3, E3, E4, A1, A2, A1d, A2d, op_A1d, op_A2d)
    ev, nv = energy_norm_4x3(C1, C2, C3, C4, E1, E2, E2, E3, E4, E4, A1, A2, A1d, A2d, op_A1d, op_A2d)

    gs_E = eh[]/nh[] + ev[]/nv[]

    return gs_E
end

function get_es_enenergy_h2(ts::CTMTensors, H)

end

function get_gs_energy_4x4(ts::CTMTensors, H)
    s1, s2, s3, s4 = H

    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es
    A = ts.A
    A1, A2, A3, A4 = A, A, A, A
    A1d, A2d, A3d, A4d = ts.Ad, ts.Ad, ts.Ad, ts.Ad

    op_A1d, op_A2d, op_A3d, op_A4d = get_op_Ad4(s1, s2, s3, s4, A1d, A2d, A3d, A4d)

    energy, norm = energy_norm_4x4(C1, C2, C3, C4, E1, E2, E3, E4, A1, A2, A3, A4, op_A1d, op_A2d, op_A3d, op_A4d)

    gs_E = energy[] / norm[]

    return gs_E
end

function get_es_energy_4x4(ts::CTMTensors, H)
    s1, s2, s3, s4 = H

    A = get_all_A(ts)
    Ad = get_all_Ad(ts)
    C1, C2, C3, C4 = get_all_Cs(ts)
    E1, E2, E3, E4 = get_all_Es(ts)
    px = get(ts.Params, "px", 0.0f0)
    py = get(ts.Params, "py", 0.0f0)

    A1, A2, A3, A4 = A, A, A, A
    A1d, A2d, A3d, A4d = Ad, Ad, Ad, Ad
    
    # for x
    E1r = shift(E1, px)
    A2, A2d = shift(A2, px), shift(A2d, px)
    A3, A3d = shift(A3, px), shift(A3d, px)
    E3r = shift(E3, px)
    C2 = shift(C2, px)
    C3 = shift(C3, px)
    E2u = shift(E2, px)
    E2d = shift(E2, px)
    
    # for y
    C4 = shift(C4, py)
    C3 = shift(C3, py)
    E3l = shift(E3, py)
    E3r = shift(E3r, py) # xxx
    E4d = shift(E4, py)
    E2d = shift(E2d, py) #xxx
    A4, A4d = shift(A4, py), shift(A4d, py)
    A3, A3d = shift(A3, py), shift(A3d, py)
    
    op_A1d, op_A2d, op_A3d, op_A4d = get_op_Ad4(s1, s2, s3, s4, A1d, A2d, A3d, A4d)

    energy, norm = energy_norm_4x4(C1, C2, C3, C4, E1, E1r, E2u, E2d, E3l, E3r, E4, E4d, A1, A2, A3, A4, A1d, A2d, A3d, A4d, op_A1d, op_A2d, op_A3d, op_A4d)

    gs_E = energy[4][] / norm[1][]

    return gs_E
end
#TODO
"""
C1 --  E1 -- E1 -- C2
|      |     |      |
E4 --  A1 -- A2 -- E2
|      |     |      |
C4 --  E3 -- E3 -- C3
"""
function energy_norm_3x4(C1, C2, C3 ,C4, E1l, E1r, E2, E3l, E3r, E4, A1, A2, A1d, A2d, op_A1d, op_A2d)

end

"""
C1 --  E1 -- C2
|      |     |
E4 --  A1 -- E2
|      |     |
E4 --  A2 -- E2
|      |     |
C4 --  E3 -- C3
"""
function energy_norm_4x3(C1, C2, C3, C4, E1, E2u, E2d, E3, E4u, E4d, A1, A2, A1d, A2d, op_A1d, op_A2d)

end

"""
'''
C1 --  E1 -- E1 -- C2
|      |     |      |
E4 --  A1 -- A2 -- E2
|      |     |      |
E4 --  A4 -- A3 -- E2  
|      |     |      |
C4 --  E3 -- E3 -- C3
```
"""
function energy_norm_4x4(C1, C2, C3, C4, E1, E2, E3, E4, A1, A2, A3, A4, op_A1d, op_A2d, op_A3d, op_A4d)
    A1d = conj(A1)
    A2d = conj(A2)
    A3d = conj(A3)
    A4d = conj(A4)
    energy_norm_4x4(C1, C2, C3, C4, E1, E1, E2, E2, E3, E3, E4, E4, A1, A2, A3, A4, A1d, A2d, A3d, A4d,op_A1d, op_A2d, op_A3d, op_A4d)
end

function energy_norm_4x4(C1, C2, C3, C4, E1l, E1r, E2u, E2d, E3l, E3r, E4u, E4d, A1, A2, A3, A4,A1d, A2d, A3d, A4d, op_A1d, op_A2d, op_A3d, op_A4d)

    C1E1 = tcon([C1, E1l], [[-1, 1], [1, -2, -3, -4]])
    CEE4 = tcon([C1E1, E4u], [[1, -2, -5, -3], [1, -1, -6, -4]])
    CEEA1 = tcon([CEE4, A1], [[-1, -2, -3, -4, 1, 2], [1, 2, -5, -6, -7]])

    energy_LU = tcon([CEEA1, op_A1d], [[-1, -4, 1, 2, -2, -5, 3], [1, 2, -3, -6, 3]])
    energy_LU = wrap_reshape(energy_LU, prod(size(energy_LU)[1:3]), :)
    norm_LU = tcon([CEEA1, A1d], [[-1, -4, 1, 2, -2, -5, 3], [1, 2, -3, -6, 3]])
    norm_LU = wrap_reshape(norm_LU, prod(size(norm_LU)[1:3]), :)

    C4E3 = tcon([C4, E3l], [[-1, 1], [1, -2, -3, -4]])
    CEE4 = tcon([C4E3, E4d], [[1, -2, -6, -4], [-1, 1, -5, -3]])
    CEEA4 = tcon([CEE4, A4], [[-1, -2, -3, -4, 1, 2], [-5, 1, 2, -6, -7]])

    energy_LB = tcon([CEEA4, op_A4d], [[-1, -4, 1, 2, -2, -5, 3], [-3, 1, 2, -6, 3]])
    energy_LB = wrap_reshape(energy_LB, prod(size(energy_LB)[1:3]), :)
    norm_LB = tcon([CEEA4, A4d], [[-1, -4, 1, 2, -2, -5, 3], [-3, 1, 2, -6, 3]])
    norm_LB = wrap_reshape(norm_LB, prod(size(norm_LB)[1:3]), :)

    C2E1 = tcon([C2, E1r], [[1, -2], [-1, 1, -3, -4]])
    CEE2 = tcon([C2E1, E2u], [[-1, 1, -5, -3], [1, -2, -6, -4]])
    CEEA2 = tcon([CEE2, A2], [[-1, -2, -3, -4, 1, 2], [1, -5, -6, 2, -7]])

    energy_RU = tcon([CEEA2, op_A2d], [[-1, -4, 1, 2, -2, -5, 3], [1, -3, -6, 2, 3]])
    energy_RU = wrap_reshape(energy_RU, prod(size(energy_RU)[1:3]), :)
    norm_RU = tcon([CEEA2, A2d], [[-1, -4, 1, 2, -2, -5, 3], [1, -3, -6, 2, 3]])
    norm_RU = wrap_reshape(norm_RU, prod(size(norm_RU)[1:3]), :)

    C3E3 = tcon([C3, E3r], [[-1, 1], [-2, 1, -3, -4]])
    CEE2 = tcon([C3E3, E2d], [[1, -2, -5, -3], [-1, 1, -6, -4]])
    CEEA3 = tcon([CEE2, A3], [[-1, -2, -5, -6, 1, 2], [-3, -4, 1, 2, -7]])

    energy_RB = tcon([CEEA3, op_A3d], [[-1, -4, -2, -5, 1, 2, 3], [-3, -6, 1, 2, 3]])
    energy_RB = wrap_reshape(energy_RB, prod(size(energy_RB)[1:3]), :)
    norm_RB = tcon([CEEA3, A3d], [[-1, -4, -2, -5, 1, 2, 3], [-3, -6, 1, 2, 3]])
    norm_RB = wrap_reshape(norm_RB, prod(size(norm_RB)[1:3]), :)

    energy_left = tcon([wrap_permutedims(energy_LU, [2, 1]),energy_LB], [[-1,1], [1,-2]])
    energy_right = tcon([energy_RU, energy_RB], [[-1,1], [1,-2]])
    # energy = tr(energy_left*energy_right)
    energy = tcon([energy_left, energy_right], [[1, 2], [1, 2]])

    norm_left = tcon([wrap_permutedims(norm_LU, [2, 1]), norm_LB], [[-1, 1], [1, -2]])
    norm_right = tcon([norm_RU, norm_RB], [[-1, 1], [1, -2]])
    # norm = tr(norm_left*norm_right)
    norm = tcon([norm_left, norm_right], [[1, 2], [1, 2]])

    energy, norm
end

"""
1
|     hamiltonian bond
2 - Ad  <
|     virtual bond
3
"""
function get_op_Ad4(s1, s2, s3, s4, A1d, A2d, A3d, A4d)
    op_A1d = tcon([s1, A1d], [[-6, 1, -4], [-1, -2, -3, -5, 1]])
    dims = size(op_A1d)
    op_A1d = wrap_reshape(op_A1d, dims[1], dims[2], dims[3], dims[4] * dims[5], dims[6])

    op_A2d = tcon([s2, A2d], [[-2, -7, 1, -4], [-1, -3, -5, -6, 1]])
    dims = size(op_A2d)
    op_A2d = wrap_reshape(op_A2d, dims[1], dims[2] * dims[3], dims[4] * dims[5], dims[6], dims[7])

    op_A3d = tcon([s3, A3d], [[-1, -7, 1, -3], [-2, -4, -5, -6, 1]])
    dims = size(op_A3d)
    op_A3d = wrap_reshape(op_A3d, dims[1] * dims[2], dims[3] * dims[4], dims[5], dims[6], dims[7])

    op_A4d = tcon([s4, A4d], [[-4, -6, 1], [-1, -2, -3, -5, 1]])
    dims = size(op_A4d)
    op_A4d = wrap_reshape(op_A4d, dims[1], dims[2], dims[3], dims[4] * dims[5], dims[6])

    return op_A1d, op_A2d, op_A3d, op_A4d
end