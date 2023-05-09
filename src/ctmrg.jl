function run_ctm(ts::CTMTensors, chi)
    min_iter = 4
    max_iter = 20
    tol = 1e-6
    diffs = [1.0]
    old_conv = 1.0
    conv = 1.0

    for i = 1:max_iter
        st_time = time()
        ts, s = rg_step(ts, chi)
        ed_time = time()

        ctm_time = ed_time - st_time

        old_conv = conv
        conv = s

        if  length(s) == length(old_conv)
            diff = norm(conv .- old_conv)
            append!(diffs, diff)
        end

        println("CTM step $(i)， conv = $(diffs[end]), time = $ctm_time")

        if i >= min_iter && diffs[end] < tol 
            println("\n ---------- CTM finished --------- \n")
            break
        end

        if i == max_iter
            println("\n --------- Not Converged ---------- \n")
        end
    end

    ts, conv
end

# function check_conv()
#     old_conv = conv
#     conv = nograd(s)

#     if  length(s) == length(old_conv)
#         diff = norm(conv .- old_conv)
#         append!(diffs, diff)
#     end

#     if i >= min_iter && diffs[end] < tol 
#         println("CTM finished, final step $(i), conv = $(diffs[end]), time = $ctm_time")
#         break
#     end

#     println("CTM step $(i)， conv = $(diffs[end]), time = $ctm_time")

#     return 
# end


function rg_step(tensors::CTMTensors, chi)
    tensors, s = left_rg(tensors, chi)
    tensors, _ = right_rg(tensors, chi)
    tensors, _ = top_rg(tensors, chi)
    tensors, _ = bottom_rg(tensors, chi)

    # tensors, _ = left_rg(tensors, chi)
    # tensors, _ = right_rg(tensors, chi)
    # tensors, _ = top_rg(tensors, chi)
    # tensors, _ = bottom_rg(tensors, chi)

    tensors, s
end

function left_rg(ts::CTMTensors, chi)
    P, Pd, s = get_projector_left(ts, chi)

    newC1, newE4, newC4 = proj_left(ts, P, Pd)

    ts = up_left(ts, newC1, newE4, newC4)

    ts, s
end

function right_rg(ts::CTMTensors, chi)
    P, Pd, s = get_projector_right(ts, chi)

    newC2, newE2, newC3 = proj_right(ts, P, Pd)

    ts = up_right(ts, newC2, newE2, newC3)

    ts, s
end

function top_rg(ts::CTMTensors, chi)
    P, Pd, s = get_projector_top(ts, chi)

    newC1, newE1, newC2 = proj_top(ts, P, Pd)

    ts = up_top(ts, newC1, newE1, newC2)

    ts, s
end

function bottom_rg(ts::CTMTensors, chi)
    P, Pd, s = get_projector_bottom(ts, chi)

    newC4, newE3, newC3 = proj_bottom(ts, P, Pd)

    ts = up_bottom(ts, newC4, newE3, newC3)

    ts, s
end

function get_projector_left(ts::CTMTensors, chi)
    A = ts.A
    Ad = ts.Ad
    C1 = ts.Cs[1]
    C4 = ts.Cs[4]
    E1 = ts.Es[1]
    E3 = ts.Es[3]
    E4 = ts.Es[4]

    tensors = [C1, C4, E1, E3, E4, E4, A, Ad, A, Ad, chi]

    get_projector_left(tensors...)
end
function get_projector_left(C1, C4, E1, E3, E4u, E4d, Au, Adu, Ad, Add, chi)
    LU = begin
        C1E1 = tcon([C1, E1], [[-1, 1], [1, -2, -3, -4]])
        CEE4 = tcon([C1E1, E4u], [[1, -2, -5, -3], [1, -1, -6, -4]])
        CEEA = tcon([CEE4, Au], [[-1, -2, -3, -4, 1, 2], [1, 2, -5, -6, -7]])
        tcon([CEEA, Adu], [[-1, -4, 1, 2, -2, -5, 3], [1, 2, -3, -6, 3]])
    end
    R1 = permutedims(reshape(LU, prod(size(LU)[1:3]), :), (2, 1))

    LB = begin
        C4E3 = tcon([C4, E3], [[-1, 1], [1, -2, -3, -4]])
        CEE4 = tcon([C4E3, E4d], [[1, -2, -6, -4], [-1, 1, -5, -3]])
        CEEA = tcon([CEE4, Ad], [[-1, -2, -3, -4, 1, 2], [-5, 1, 2, -6, -7]])
        tcon([CEEA, Add], [[-1, -4, 1, 2, -2, -5, 3], [-3, 1, 2, -6, 3]])
    end
    R2 = reshape(LB, prod(size(LB)[1:3]), :)

    get_projector(R1, R2, chi)
end

function get_projector_right(ts::CTMTensors, chi)
    A = ts.A
    Ad = ts.Ad
    C2 = ts.Cs[2]
    C3 = ts.Cs[3]
    E1 = ts.Es[1]
    E2 = ts.Es[2]
    E3 = ts.Es[3]

    tensors = [C2, C3, E1, E2, E2, E3, A, Ad, A, Ad, chi]

    get_projector_right(tensors...)
end
function get_projector_right(C2, C3, E1, E2u, E2d, E3, Au, Adu, Ad, Add, chi)
    RU = begin
        C2E1 = tcon([C2, E1], [[1, -2], [-1, 1, -3, -4]])
        CEE2 = tcon([C2E1, E2u], [[-1, 1, -5, -3], [1, -2, -6, -4]])
        CEEA = tcon([CEE2, Au], [[-1, -2, -3, -4, 1, 2], [1, -5, -6, 2, -7]])
        tcon([CEEA, Adu], [[-1, -4, 1, 2, -2, -5, 3], [1, -3, -6, 2, 3]])
    end
    R1 = reshape(RU, prod(size(RU)[1:3]), :)

    RB = begin
        C3E3 = tcon([C3, E3], [[-1, 1], [-2, 1, -3, -4]])
        CEE2 = tcon([C3E3, E2d], [[1, -2, -5, -3], [-1, 1, -6, -4]])
        CEEA = tcon([CEE2, Ad], [[-1, -2, -5, -6, 1, 2], [-3, -4, 1, 2, -7]])
        tcon([CEEA, Add], [[-1, -4, -2, -5, 1, 2, 3], [-3, -6, 1, 2, 3]])
    end
    R2 = reshape(RB, prod(size(RB)[1:3]), :)

    get_projector(R1, R2, chi)
end

function get_projector_top(ts::CTMTensors, chi)
    A = ts.A
    Ad = ts.Ad
    C1 = ts.Cs[1]
    C2 = ts.Cs[2]
    E1 = ts.Es[1]
    E2 = ts.Es[2]
    E4 = ts.Es[4]

    tensors = [C1, C2, E1, E1, E2, E4, A, Ad, A, Ad, chi]

    get_projector_top(tensors...)
end
function get_projector_top(C1, C2, E1l, E1r, E2, E4, Al, Adl, Ar, Adr, chi)
    UL = begin
        C1E4 = tcon([C1, E4], [[1, -2], [1, -1, -3, -4]])
        CEE1 = tcon([C1E4, E1l], [[-1, 1, -6, -4], [1, -2, -5, -3]])
        CEEA = tcon([CEE1, Al], [[-1, -2, -5, -6, 1, 2], [1, 2, -3, -4, -7]])
        tcon([CEEA, Adl], [[-1, -4, -2, -5, 1, 2, 3], [1, 2, -3, -6, 3]])
    end
    R1 = reshape(UL, prod(size(UL)[1:3]), :)

    UR = begin
        C2E2 = tcon([C2, E2], [[-1, 1], [1, -2, -3, -4]])
        CEE1 = tcon([C2E2, E1r], [[1, -2, -6, -4], [-1, 1, -5, -3]])
        CEEA = tcon([CEE1, Ar], [[-1, -2, -5, -6, 1, 2], [1, -3, -4, 2, -7]])
        tcon([CEEA, Adr], [[-1, -4, -2, -5, 1, 2, 3], [1, -3, -6, 2, 3]])
    end
    R2 = reshape(UR, prod(size(UR)[1:3]), :)

    get_projector(R1, R2, chi)
end

function get_projector_bottom(ts::CTMTensors, chi)
    A = ts.A
    Ad = ts.Ad
    C3 = ts.Cs[3]
    C4 = ts.Cs[4]
    E2 = ts.Es[2]
    E3 = ts.Es[3]
    E4 = ts.Es[4]

    tensors = [C3, C4, E2, E3, E3, E4, A, Ad, A, Ad, chi]

    get_projector_bottom(tensors...)
end
function get_projector_bottom(C3, C4, E2, E3l, E3r, E4, Al, Adl, Ar, Adr, chi)
    BL = begin
        C4E4 = tcon([C4, E4], [[1, -2], [-1, 1, -3, -4]])
        CEE3 = tcon([C4E4, E3l], [[-1, 1, -5, -3], [1, -2, -6, -4]])
        CEEA = tcon([CEE3, Al], [[-1, -2, -5, -6, 1, 2], [-3, 1, 2, -4, -7]])
        tcon([CEEA, Adl], [[-1, -4, -2, -5, 1, 2, 3], [-3, 1, 2, -6, 3]])
    end
    R1 = reshape(BL, prod(size(BL)[1:3]), :)

    BR = begin
        C3E2 = tcon([C3, E2], [[1, -2], [-1, 1, -3, -4]])
        CEE3 = tcon([C3E2, E3r], [[-1, 1, -6, -4], [-2, 1, -5, -3]])
        CEEA = tcon([CEE3, Ar], [[-1, -2, -5, -6, 1, 2], [-3, -4, 1, 2, -7]])
        tcon([CEEA, Adr], [[-1, -4, -2, -5, 1, 2, 3], [-3, -6, 1, 2, 3]])
    end
    R2 = permutedims(reshape(BR, prod(size(BR)[1:3]), :), (2, 1))

    get_projector(R1, R2, chi)
end

function get_projector(R1, R2, chi)
    # BUG: potentional; should be size(R2,2)
    new_chi = min(chi, size(R1, 2))

    U, S, V = svd(R1 * R2)
    ####### cut off
    new_chi = count(>=(S[new_chi] - 1.0E-10), S)
    U1 = U[:, 1:new_chi]
    V1 = V[:, 1:new_chi]
    # S = S./S[1]
    S1 = S[1:new_chi]

    # display(S1)
    # cut_off = sum(S[new_chi+1:end]) / sum(S)   

    inv_sqrt_S = sqrt.(S1) |> diag_inv #|> diagm |> inv

    P1 = R2 * V1 * inv_sqrt_S    #(D2,D3)*(D3,chi)*(chi,chi) --> (D2, chi) 
    P2 = inv_sqrt_S * U1' * R1   #(chi,chi)*(chi,D1)*(D1,D2)  --> (chi, D2)

    P1, P2, S1
end

function proj_left(ts, P, Pd)
    # A    = get_all_A(ts)
    # Ad   = get_all_Ad(ts)
    # C1, C2, C3, C4 = get_all_Cs(ts)
    # E1, E2, E3, E4 = get_all_Es(ts)
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

    D = size(A, 1)

    newC1 = begin
        C1E1 = tcon([C1, E1], [[-2, 1], [1, -1, -3, -4]])
        CEP = reshape(C1E1, size(C1E1, 1), :) * P
        permutedims(CEP, (2,1))
    end
    newC4 = begin
        C4E3 = tcon([C4, E3], [[-1, 1], [1, -4, -2, -3]])
        Pd * reshape(C4E3, :, size(C4E3, 4))
    end

    newE4 = begin
        E4A = tcon([E4, A], [[-1, -3, 1, -6], [-2, 1, -4, -5, -7]])
        EAAd = tcon([E4A, Ad], [[-1, -2, -4, -5, -7, 1, 2], [-3, 1, -6, -8, 2]])
        EAAd = reshape(EAAd, prod(size(EAAd)[1:3]), prod(size(EAAd)[4:6]), D, D)
        EPd = tcon([EAAd, Pd], [[1, -2, -3, -4], [-1, 1]])
        tcon([EPd, P], [[-1, 1, -3, -4], [1, -2]])
    end

    renormalize(newC1), renormalize(newE4), renormalize(newC4)
end

function proj_right(ts, P, Pd)
    # A    = get_all_A(ts)
    # Ad   = get_all_Ad(ts)
    # C1, C2, C3, C4 = get_all_Cs(ts)
    # E1, E2, E3, E4 = get_all_Es(ts)
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

    D = size(A, 1)

    newC2 = begin
        C2E1 = tcon([C2, E1], [[1, -2], [-1, 1, -3, -4]])
        reshape(C2E1, size(C2E1, 1), :) * P
    end
    newC3 = begin
        C3E3 = tcon([C3, E3], [[-1, 1], [-4, 1, -2, -3]])
        Pd * reshape(C3E3, :, size(C3E3, 4))
    end

    newE2 = begin
        E2A = tcon([E2, A], [[-1, -3, 1, -6], [-2, -5, -4, 1, -7]])
        EAAd = tcon([E2A, Ad], [[-1, -2, -4, -5, -7, 1, 2], [-3, -8, -6, 1, 2]])
        EAAd = reshape(EAAd, prod(size(EAAd)[1:3]), prod(size(EAAd)[4:6]), D, D)
        EPd = tcon([EAAd, Pd], [[1, -2, -3, -4], [-1, 1]])
        tcon([EPd, P], [[-1, 1, -3, -4], [1, -2]])
    end

    renormalize(newC2), renormalize(newE2), renormalize(newC3)
end

function proj_top(ts, P, Pd)
    # A    = get_all_A(ts)
    # Ad   = get_all_Ad(ts)
    # C1, C2, C3, C4 = get_all_Cs(ts)
    # E1, E2, E3, E4 = get_all_Es(ts)
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

    D = size(A, 1)

    newC1 = begin
        C1E4 = tcon([C1, E4], [[1, -2], [1, -1, -3, -4]])
        reshape(C1E4, size(C1E4, 1), :) * P
    end

    newC2 = begin
        C2E2 = tcon([C2, E2], [[-1, 1], [1, -4, -2, -3]])
        Pd * reshape(C2E2, :, size(C2E2, 4))
    end

    newE1 = begin
        E1A = tcon([E1, A], [[-1, -3, 1, -6], [1, -2, -5, -4, -7]])
        EAAd = tcon([E1A, Ad], [[-1, -2, -4, -5, -7, 1, 2], [1, -3, -8, -6, 2]])
        EAAd = reshape(EAAd, prod(size(EAAd)[1:3]), prod(size(EAAd)[4:6]), D, D)
        EPd = tcon([EAAd, Pd], [[1, -2, -3, -4], [-1, 1]])
        tcon([EPd, P], [[-1, 1, -3, -4], [1, -2]])
    end

    renormalize(newC1), renormalize(newE1), renormalize(newC2)
end

function proj_bottom(ts, P, Pd)
    # A    = get_all_A(ts)
    # Ad   = get_all_Ad(ts)
    # C1, C2, C3, C4 = get_all_Cs(ts)
    # E1, E2, E3, E4 = get_all_Es(ts)
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

    D = size(A, 1)

    newC4 = begin
        C4E4 = tcon([C4, E4], [[1, -2], [-1, 1, -3, -4]])
        reshape(C4E4, size(C4E4, 1), :) * P
    end

    newC3 = begin
        C3E2 = tcon([C3, E2], [[1, -1], [-4, 1, -2, -3]])
        CEP = Pd * reshape(C3E2, :, size(C3E2, 4))
        permutedims(CEP, (2,1))
    end

    newE3 = begin
        E3A = tcon([E3, A], [[-1, -3, 1, -6], [-5, -2, 1, -4, -7]])
        EAAd = tcon([E3A, Ad], [[-1, -2, -4, -5, -7, 1, 2], [-8, -3, 1, -6, 2]])
        EAAd = reshape(EAAd, prod(size(EAAd)[1:3]), prod(size(EAAd)[4:6]), D, D)
        EPd = tcon([EAAd, Pd], [[1, -2, -3, -4], [-1, 1]])
        tcon([EPd, P], [[-1, 1, -3, -4], [1, -2]])
    end

    renormalize(newC4), renormalize(newE3), renormalize(newC3)
end

function up_left(ts, C1, E4, C4)
    if C1 isa NestedTensor
        # @set ts.Cs[1]
    else 
        newCs = [C1, ts.Cs[2], ts.Cs[3], C4]
        newEs = [ts.Es[1], ts.Es[2], ts.Es[3], E4]
        ts = setproperties(ts, Cs = newCs, Es = newEs)
    end

    ts
end

function up_right(ts, C2, E2, C3)
    if C2 isa NestedTensor
        # @set ts.Cs[1]
    else 
        newCs = [ts.Cs[1], C2, C3, ts.Cs[4]]
        newEs = [ts.Es[1], E2, ts.Es[3], ts.Es[4]]
        ts = setproperties(ts, Cs = newCs, Es = newEs)
    end

    ts
end

function up_top(ts, C1, E1, C2)
    if C1 isa NestedTensor
        # @set ts.Cs[1]
    else 
        newCs = [C1, C2, ts.Cs[3], ts.Cs[4]]
        newEs = [E1, ts.Es[2], ts.Es[3], ts.Es[4]]
        ts = setproperties(ts, Cs = newCs, Es = newEs)
    end

    ts
end

function up_bottom(ts, C4, E3, C3)
    if C4 isa NestedTensor
        # @set ts.Cs[1]
    else 
        newCs = [ts.Cs[1], ts.Cs[2], C3, C4]
        newEs = [ts.Es[1], ts.Es[2], E3, ts.Es[4]]
        ts = setproperties(ts, Cs = newCs, Es = newEs)
    end

    ts
end