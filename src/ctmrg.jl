
function rg_step(tensors::CTMTensors, chi)
    tensors, s = left_rg(tensors, chi)
    tensors, _ = right_rg(tensors, chi)
    tensors, _ = top_rg(tensors, chi)
    tensors, _ = bottom_rg(tensors, chi)

    tensors, s
end

function left_rg(ts::CTMTensors, chi)
    Pl, Pld, s = get_projector_left(ts, chi)

    newC1, newE4, newC4 = proj_left(ts, Pl, Pld)

    ts = up_left(ts, newC1, newE4, newC4)
    ts, s
end

function get_projector_left(ts::CTMTensors, chi)
    A    = ts.A
    Ad   = ts.Ad
    C1   = ts.Cs[1]
    C4   = ts.Cs[4]
    E1   = ts.Es[1]
    E3   = ts.Es[3]
    E4   = ts.Es[4]

    tensors = [C1, C4, E1, E3, E4, E4, A, Ad, A, Ad, chi]

    get_projector_left(tensors...)
end
function get_projector_left(C1,C4,E1,E3,E4u,E4d,Au,Adu,Ad,Add, chi)
    UL = begin
        C1E1 = tcon([C1, E1], [[-1,1], [1-2,-3,-4]])
        CEE4 = tcon([C1E1, E4u], [[1,-2,-5,-3], [1,-1,-6,-4]])
        CEEA = tcon([CEE4, Au], [[-1,-2,-3,-4,1,2], [1,2,-5,-6,-7]])
        tcon([CEEA, Adu], [-1,-4,1,2,-2,-5,3], [1,2,-3,-6,3])
    end
    R1 = permutedims(reshape(UL, sum(size(UL)[1:3]), :), (2,1))

    BL = begin
        C4E3 = tcon([C4,E3], [[-1,1], [1,-2,-3,-4]])
        CEE4 = tcon([C4E3, E4d], [[1,-2,-6,-4],[-1,1,-5,-3]])
        CEEA = tcon([CEE4, Ad], [[-1,-2,-3,-4,1,2], [-5,1,2,-6,-7]])
        tcon([CEEA, Add], [[-1,-4,1,2,-2,-5,3], [-3,1,2,-6,3]])
    end
    R2 = reshape(BL, sum(size(BL)[1:3]), :)

    get_projector(R1,R2, chi)
end

function get_projector_right(ts::CTMTensors, chi)
    A    = ts.A
    Ad   = ts.Ad
    C2   = ts.Cs[2]
    C3   = ts.Cs[3]
    E1   = ts.Es[1]
    E2   = ts.Es[2]
    E3   = ts.Es[3]

    tensors = [C2, C3, E1, E2, E2, E3, A, Ad, A, Ad, chi]

    get_projector_right(tensors...)
end
function get_projector_right(C2,C3,E1,E2u,E2d,E3,Au,Adu,Ad,Add, chi)
    UR =begin
        C2E1 = tcon([C2, E1], [[1,-2],[-1, 1, -3, -4]])
        CEE2 = tcon([C2E1, E2u], [[-1,1,-5,-3], [1,-2,-6,-4]])
        CEEA = tcon([CEE2, Au], [[-1,-2,-3,-4,1,2],[1,-5,-6,2,-7]])
        tcon([CEEA, Adu], [[-1,-4,1,2,-2,-5,3],[1,-3,-6,2,3]])
    end
end

function get_projector_top()
    A    = ts.A
    Ad   = ts.Ad
    C1   = ts.Cs[1]
    C2   = ts.Cs[2]
    E1   = ts.Es[1]
    E2   = ts.Es[2]
    E4   = ts.Es[4]

    tensors = [C1, C2, E1, E1, E2, E4, A, Ad, A, Ad, chi]

    get_projector_top(tensors...)
end
function get_projector_top(C1,C2,T1l,T1r,T2,T4,Al,Adl,Ar,Adr, chi)
    
end

function get_projector_bottom()
    A    = ts.A
    Ad   = ts.Ad
    C3   = ts.Cs[3]
    C4   = ts.Cs[4]
    E2   = ts.Es[2]
    E3   = ts.Es[3]
    E4   = ts.Es[4]

    tensors = [C3, C4, E2, E3, E3, E4, A, Ad, A, Ad, chi]

    get_projector_bottom(tensors...)
end
function get_projector_bottom(C3,C4,T2,T3l,T3r,T4,Al,Adl,Ar,Adr, chi)
    
end

function get_projector(R1, R2, chi)
    # BUG: potentional; should be size(R2,2)
    new_chi = min(chi, size(R1,2)) 

    U, S, V = svd(R1*R2)
    ####### cut off
    new_chi = count(>=(S[new_chi]-1.0E-12),S) 
    U1 = U[:, 1:new_chi]
    V1 = V[:, 1:new_chi]
    # S = S./S[1]
    S1 = S[1:new_chi]
    
    
    # cut_off = sum(S[new_chi+1:end]) / sum(S)   

    inv_sqrt_S = sqrt.(S1) |> diagm |> inv

    P1 = R2*V1*inv_sqrt_S    #(D2,D3)*(D3,chi)*(chi,chi) --> (D2, chi) 
    P2 = inv_sqrt_S*U1'*R1   #(chi,chi)*(chi,D1)*(D1,D2)  --> (chi, D2)

    P1, P2, S1
end
