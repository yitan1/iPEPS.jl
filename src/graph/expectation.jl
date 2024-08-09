function compute_es(px, py, filename::String; disp=false)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)
    compute_es(px, py, cfg; disp=disp)
end

function compute_es(px, py, cfg::Dict; disp=false)
    es_name = get_es_name(cfg, px, py)
    es_file = load(es_name)
    effH = es_file["effH"]
    effN = es_file["effN"]
    fprint("load H and N at $es_name")

    nrmB_cut = get(cfg, "nrmB_cut", 1e-3)

    H = (effH + effH') / 2
    N = (effN + effN') / 2
    ev_N, P = eigen(N)
    idx = sortperm(real.(ev_N))[end:-1:1]
    ev_N = ev_N[idx]
    if nrmB_cut isa Int
        selected = ev_N .> ev_N[nrmB_cut+1]
    else
        selected = (ev_N / maximum(ev_N)) .> nrmB_cut
    end

    if disp == true
        display(ev_N / maximum(ev_N))
        display(ev_N[selected] / maximum(ev_N))
    end

    P = P[:, idx]
    P = P[:, selected]
    N2 = P' * N * P
    H2 = P' * H * P
    H2 = (H2 + H2') / 2
    N2 = (N2 + N2') / 2
    es, vecs = eigen(H2, N2)
    ixs = sortperm(real.(es))
    es = es[ixs]
    vecs = vecs[:, ixs]

    if haskey(es_file, "envB")
        envB = es_file["envB"]
        println("load envB")
        return es, vecs, P, envB
    else
        return es, vecs, P
    end
end

function compute_spec_env(op, px, py, filename::String; proj=true)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)

    basis_name = get_basis_name(cfg)
    basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")

    ts = setproperties(ts, Params=cfg)
    ts.Params["px"] = convert(eltype(ts.A), px * pi)
    ts.Params["py"] = convert(eltype(ts.A), py * pi)

    B = tcon([ts.A, op], [[-1, -2, -3, -4, 1], [-5, 1]])
    if proj
        C1, C2, C3, C4 = ts.Cs
        E1, E2, E3, E4 = ts.Es
        n_dm = iPEPS.get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)
        ndm_Ad = iPEPS.tcon([n_dm, ts.Ad], [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5]])
        nAA = transpose(ts.A[:]) * ndm_Ad[:]

        B = B[:] .- (transpose(B[:]) * ndm_Ad[:]) * ts.A[:] ./ nAA
        B = reshape(B, size(ts.A))
    end
    # B = reshape(basis[:,16], size(ts.A)) 
    # Bd = conj(B)

    ts = setproperties(ts, B=B, Bd=conj(B))

    # conv_fun(_x) = get_es_energy(_x, H) / get_all_norm(_x)[1]
    ts, _ = run_ctm(ts)

    _, envB = get_all_norm(ts)

    envB, basis
end

function lor_broad(x, es, swk, factor)
    w = 0.0
    for i in eachindex(swk)
        w += 1 / pi * factor / ((x - es[i])^2 + factor^2) * swk[i]
    end

    w
end

function compute_B_env(B, px, py, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)


    basis_name = get_basis_name(cfg)
    # basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")
    H = load(basis_name, "H")

    ts = setproperties(ts, Params=cfg)
    ts.Params["px"] = convert(eltype(ts.A), px * pi)
    ts.Params["py"] = convert(eltype(ts.A), py * pi)

    ts = setproperties(ts, B=B, Bd=conj(B))

    conv_fun(_x) = get_es_energy(_x, H) / get_all_norm(_x)[1]
    ts, _ = run_ctm(ts, conv_fun=conv_fun)

    _, envB = get_all_norm(ts)

    envB
end


function evaluate_es(px, py, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load dafault config file")
    end
    print_cfg(cfg)

    evaluate_es(px, py, cfg)
end
function evaluate_es(px, py, cfg::Dict)
    es_name = get_es_name(cfg, px, py)
    effH = load(es_name, "effH")
    effN = load(es_name, "effN")
    fprint("load H and N at $es_name")

    nrmB_cut = get(cfg, "nrmB_cut", 1e-3)

    H = (effH + effH') / 2
    N = (effN + effN') / 2
    ev_N, P = eigen(N)
    idx = sortperm(real.(ev_N))[end:-1:1]
    ev_N = ev_N[idx]
    # display(ev_N)
    if nrmB_cut isa Int
        selected = ev_N .> ev_N[nrmB_cut+1]
    else
        selected = (ev_N / maximum(ev_N)) .> nrmB_cut
    end
    P = P[:, idx]
    P = P[:, selected]
    N2 = P' * N * P
    H2 = P' * H * P
    H2 = (H2 + H2') / 2
    N2 = (N2 + N2') / 2
    es, vecs = eigen(H2, N2)
    ixs = sortperm(real.(es))
    es = es[ixs]
    vecs = vecs[:, ixs]

    es, vecs
end

function correlation_TM_A(A, cfg::Dict; direction="h")
    ts0 = CTMTensors(A, cfg)
    println("get converged boundary CTM")
    ts0, _ = run_ctm(ts0)

    return correlation_TM(ts0, direction=direction)
end

function correlation_TM(ts0; direction="h", N=10, full=false)
    # C1, C2, C3, C4 = ts0.Cs
    E1, E2, E3, E4 = ts0.Es
    # A = ts0.A
    # Ad = ts0.Ad

    if direction == "h"
        TM = tcon([E4, E2], [[-1, -3, 1, 2], [-2, -4, 1, 2]])
        # @ein T[m1, m2, m3, m4, m5, m6, m7, m8] := A[m1, m3, m5, m7, p1] * Ad[m2, m4, m6, m8, p1]
        # @ein TM[m1, m2, m3, m4, m5, m6, m7, m8] := (E1[m1, m5, p1, p2] * T[p1, p2, m2, m3, p3, p4, m6, m7]) * E3[m4, m8, p3, p4]
        # TM = reshape(TM, prod(size(TM)[1:4]), :)
    elseif direction == "v"
        TM = tcon([E1, E3], [[-1, -3, 1, 2], [-2, -4, 1, 2]])
    end

    TM = reshape(TM, prod(size(TM)[1:2]), :)

    # TM = (TM' .+ TM) ./2
    if full
        es, _ = eigen(TM)
    else
        es, _ = eigsolve(TM, N, :LM)
    end
    es1 = abs.(es)
    es1 = es1[sortperm(es1)[end:-1:1]]

    return es1[es1.>0], es, TM
end

function correlation_spin_A(A, op, cfg::Dict; max_iter=10, direction="h")
    correlation_spin_A(A, op, op, cfg, max_iter=max_iter, direction=direction)
end

function correlation_spin_A(A, op1, op2, cfg::Dict; max_iter=10, direction="h")
    ts0 = CTMTensors(A, cfg)
    println("get converged boundary CTM")
    ts0, _ = run_ctm(ts0)

    return correlation_spin(ts0, op1, op2, max_iter=max_iter, direction=direction)
end

function correlation_spin(op, cfg::Dict; max_iter=10, direction="h")
    correlation_spin(op, op, cfg, max_iter=max_iter, direction=direction)
end

function correlation_spin(op1, op2, cfg::Dict; max_iter=10, direction="h")
    bs_name = get_basis_name(cfg)
    ts0 = load(bs_name, "ts")
    println("load $bs_name")

    correlation_spin(ts0, op1, op2, max_iter=max_iter, direction=direction)
end

function correlation_spin(ts0::CTMTensors, op1, op2; max_iter=10, direction="h")
    if direction == "h"
        Et, Ed, T, T_op1, T_op2, L0, R0 = get_envs_TM_h(ts0, op1, op2)
    else
        Et, Ed, T, T_op1, T_op2, L0, R0 = get_envs_TM_v(ts0, op1, op2)
    end
    L = con_LT(Et, Ed, L0, T_op1, direction=direction)
    R = con_TR(Et, Ed, T_op2, R0, direction=direction)

    s2s = [con_LR(L, R)]
    println("r = 1, correlation: ", s2s[end])

    m1 = con_LR(L, R0)
    m2 = con_LR(L0, R)
    for r = 1:max_iter-1
        L = con_LT(Et, Ed, L, T, direction=direction)
        exp_val = con_LR(L, R)

        append!(s2s, exp_val)

        println("r = $(r+1), correlation: ", s2s[end])
        flush(stdout)
    end

    s2s, m1, m2
end

function con_LT(Et, Ed, L0, T; direction="h")
    if direction == "h"
        @ein L[p1, p2, p3, p4] := ((L0[m1, m4, m5, m8] * Et[m1, p1, m2, m3]) * T[m2, m3, m4, m5, m6, m7, p2, p3]) * Ed[m8, p4, m6, m7]
    elseif direction == "v"
        @ein L[p1, p2, p3, p4] := ((L0[m1, m4, m5, m8] * Et[m1, p1, m2, m3]) * T[m4, m5, m2, m3, p2, p3, m6, m7]) * Ed[m8, p4, m6, m7]
    end

    return L
end

function con_TR(Et, Ed, T, R0; direction="h")
    if direction == "h"
        @ein R[p1, p2, p3, p4] := ((R0[m1, m4, m5, m8] * Et[p1, m1, m2, m3]) * T[m2, m3, p2, p3, m6, m7, m4, m5]) * Ed[p4, m8, m6, m7]
    elseif direction == "v"
        @ein R[p1, p2, p3, p4] := ((R0[m1, m4, m5, m8] * Et[p1, m1, m2, m3]) * T[p2, p3, m2, m3, m4, m5, m6, m7]) * Ed[p4, m8, m6, m7]
    end

    return R
end

function con_LR(L, R)
    @ein lr[] := L[m1, m2, m3, m4] * R[m1, m2, m3, m4]

    return lr[]
end

function get_envs_TM_h(ts0, op1, op2)
    C1, C2, C3, C4 = ts0.Cs
    E1, E2, E3, E4 = ts0.Es
    A = ts0.A
    Ad = ts0.Ad

    ## C1, E4, C4
    @ein L0[m1, m2, m3, m4] := (C1[p1, m1] * E4[p1, p2, m2, m3]) * C4[p2, m4]
    # L0 = L0[:]

    # C2, E2, C3
    @ein R0[m1, m2, m3, m4] := (C2[m1, p1] * E2[p1, p2, m2, m3]) * C3[p2, m4]
    # R0 = R0[:]

    # E1, A, op, Ad, E3
    @ein T[m1, m2, m3, m4, m5, m6, m7, m8] := A[m1, m3, m5, m7, p1] * Ad[m2, m4, m6, m8, p1]
    @ein T_op1[m1, m2, m3, m4, m5, m6, m7, m8] := (A[m1, m3, m5, m7, p1] * op1[p1, p2]) * Ad[m2, m4, m6, m8, p2]
    @ein T_op2[m1, m2, m3, m4, m5, m6, m7, m8] := (A[m1, m3, m5, m7, p1] * op2[p1, p2]) * Ad[m2, m4, m6, m8, p2]

    # @ein TM[m1, m2, m3, m4, m5, m6, m7, m8] := (E1[m1, m5, p1, p2] * T[p1, p2, m2, m3, p3, p4, m6, m7]) * E3[m4, m8, p3, p4]
    # @ein TM_op1[m1, m2, m3, m4, m5, m6, m7, m8] := (E1[m1, m5, p1, p2] * T_op1[p1, p2, m2, m3, p3, p4, m6, m7]) * E3[m4, m8, p3, p4]
    # @ein TM_op2[m1, m2, m3, m4, m5, m6, m7, m8] := (E1[m1, m5, p1, p2] * T_op2[p1, p2, m2, m3, p3, p4, m6, m7]) * E3[m4, m8, p3, p4]

    # TM = reshape(TM, prod(size(TM)[1:4]), :)
    # TM_op1 = reshape(TM_op1, prod(size(TM_op1)[1:4]), :)
    # TM_op2 = reshape(TM_op2, prod(size(TM_op2)[1:4]), :)
    Et = E1
    Ed = E3

    return Et, Ed, T, T_op1, T_op2, L0, R0
end

function get_envs_TM_v(ts0, op1, op2)
    C1, C2, C3, C4 = ts0.Cs
    E1, E2, E3, E4 = ts0.Es
    A = ts0.A
    Ad = ts0.Ad

    ## C1, E1, C2
    @ein U0[m1, m2, m3, m4] := (C1[m1, p1] * E1[p1, p2, m2, m3]) * C2[p2, m4]
    # U0 = U0[:]

    # C4, E3, C3
    @ein D0[m1, m2, m3, m4] := (C4[m1, p1] * E3[p1, p2, m2, m3]) * C3[m4, p2]
    # D0 = D0[:]

    # E4, A, op, Ad, E2
    @ein T[m1, m2, m3, m4, m5, m6, m7, m8] := A[m1, m3, m5, m7, p1] * Ad[m2, m4, m6, m8, p1]
    @ein T_op1[m1, m2, m3, m4, m5, m6, m7, m8] := (A[m1, m3, m5, m7, p1] * op1[p1, p2]) * Ad[m2, m4, m6, m8, p2]
    @ein T_op2[m1, m2, m3, m4, m5, m6, m7, m8] := (A[m1, m3, m5, m7, p1] * op2[p1, p2]) * Ad[m2, m4, m6, m8, p2]

    # @ein TM[m1, m2, m3, m4, m5, m6, m7, m8] := (E4[m1, m5, p1, p2] * T[m2, m3, p1, p2, m6, m7, p3, p4]) * E2[m4, m8, p3, p4]
    # @ein TM_op1[m1, m2, m3, m4, m5, m6, m7, m8] := (E4[m1, m5, p1, p2] * T_op1[m2, m3, p1, p2, m6, m7, p3, p4]) * E2[m4, m8, p3, p4]
    # @ein TM_op2[m1, m2, m3, m4, m5, m6, m7, m8] := (E4[m1, m5, p1, p2] * T_op2[m2, m3, p1, p2, m6, m7, p3, p4]) * E2[m4, m8, p3, p4]

    # TM = reshape(TM, prod(size(TM)[1:4]), :)
    # TM_op1 = reshape(TM_op1, prod(size(TM_op1)[1:4]), :)
    # TM_op2 = reshape(TM_op2, prod(size(TM_op2)[1:4]), :)

    El = E4
    Er = E2
    return El, Er, T, T_op1, T_op2, U0, D0
end

# function correlation_spin(ts0::CTMTensors, op1, op2; step=1, max_iter=10, direction="l")
#     ts = get_spin_boundary(op2, ts0, direction=direction)
#     ts.Params["max_iter"] = step
#     s2s = []
#     ss = []
#     for _ = 1:max_iter
#         ts, _ = run_ctm(ts)
#         ndm_Ad = get_env_A(ts)
#         # compute A * S * n_dm_Ad
#         A_op = tcon([ts0.A, op1], [[-1, -2, -3, -4, 1], [1, -5]])

#         ss0 = tcon([ndm_Ad, A_op], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
#         s0 = tcon([ndm_Ad, ts0.A], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
#         append!(s2s, ss0[])
#         append!(ss, s0[])

#         println("correlation: ", s2s[end])
#         println("single operator: ", ss[end])
#     end

#     s2s, ss
# end

# function get_spin_boundary(op, ts0; direction="l")
#     A_op = tcon([ts0.A, op], [[-1, -2, -3, -4, 1], [1, -5]])

#     chi = ts0.Params["chi"]
#     A0 = ts0.A
#     ts = ts0

#     if direction == "l" 
#         ts = setproperties(ts, A=A_op)
#         ts, _ = left_rg(ts, chi)
#         ts = setproperties(ts, A=A0)
#     else 
#         ts, _ = left_rg(ts, chi)
#     end

#     if direction == "r"
#         ts = setproperties(ts, A=A_op)
#         ts, _ = right_rg(ts, chi)
#         ts = setproperties(ts, A=A0)
#     else 
#         ts, _ = right_rg(ts, chi)
#     end

#     if direction == "u"
#         ts = setproperties(ts, A=A_op)
#         ts, _ = top_rg(ts, chi)
#         ts = setproperties(ts, A=A0)
#     else
#         ts, _ = top_rg(ts, chi)
#     end

#     if direction == "d"
#         ts = setproperties(ts0, A=A_op)
#         ts, _ = bottom_rg(ts, chi)
#         ts = setproperties(ts, A=A0)
#     else
#         ts, _ = bottom_rg(ts, chi)
#     end

#     ts
# end

# function get_env_A(ts)
#     Ad = ts.Ad
#     C1, C2, C3, C4 = ts.Cs
#     E1, E2, E3, E4 = ts.Es
#     n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)
#     ndm_Ad = tcon([n_dm, Ad], [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5]])
#     # nrm0 = tcon([ndm_Ad, A], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

#     return ndm_Ad
# end