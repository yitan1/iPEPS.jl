function evaluate_wp(filename::String; disp = true)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load dafault config file")
    end
    print_cfg(cfg)

    evaluate_wp(cfg; disp = disp)
end
function evaluate_wp(cfg::Dict; disp = true)
    wp_name = get_wp_name(cfg)
    effH = load(wp_name, "effH")
    effN = load(wp_name, "effN")
    fprint("load H and N at $wp_name")

    nrmB_cut = get(cfg, "nrmB_cut", 1e-3)

    H = (effH + effH') /2 
    N = (effN + effN') /2
    ev_N, P = eigen(N)
    idx = sortperm(real.(ev_N))[end:-1:1]
    ev_N = ev_N[idx]
    if nrmB_cut isa Int
        selected = ev_N .> ev_N[nrmB_cut+1]
    else
        selected = (ev_N/maximum(ev_N) ) .> nrmB_cut
    end
    if disp == true
        display(ev_N/maximum(ev_N))
        display(ev_N[selected] /maximum(ev_N))
    end
    P = P[:,idx]
    P = P[:,selected]
    N2 = P' * N * P
    H2 = P' * H * P
    H2 = (H2 + H2') /2 
    N2 = (N2 + N2') /2
    es, vecs = eigen(H2,N2)
    ixs = sortperm(real.(es))
    es = es[ixs]
    vecs = vecs[:,ixs]

    es, vecs, P
end

function optim_wp(filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)

    optim_wp(cfg)
end

function optim_wp(cfg)
    basis_name = get_basis_name(cfg)
    basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")
    ts = setproperties(ts, Params = cfg)
    fprint("load basis, ts in $basis_name")

    basis = complex(basis) # ！！！！ convert Complex
    basis_dim = size(basis, 2) 

    wp_name = get_wp_name(ts.Params)
    effH = zeros(ComplexF64, basis_dim, basis_dim)
    effN = zeros(ComplexF64, basis_dim, basis_dim)

    for i = 1:basis_dim
        @time gH, gN = get_wp_grad(ts, basis[:,i])
        effH[:, i] = transpose(conj(basis)) * gH / 2
        effN[:, i] = transpose(conj(basis)) * gN / 2

        fprint("\nFinish basis vector of $(i)/$(basis_dim)")

        if ts.Params["save"]
            jldsave(wp_name; effH = effH, effN = effN)
            fprint("Saved (effH, effN) and envB to $(wp_name)")
        end

        if ts.Params["gc"]
            GC.gc()
        end
    end

    if ts.Params["save"]
        jldsave(wp_name; effH = effH, effN = effN)
        fprint("Saved (effH, effN) and envB to $(wp_name)")
    end

    if ts.Params["gc"]
        GC.gc()
    end

    effH, effN
end

function get_wp_grad(ts, Bi)
    B = reshape(Bi, size(ts.A))
    # Bd = conj(B)

    # (e, n), back = Zygote.pullback(x -> run_wp_exact(ts, x), B)
    if ts.Params["wp"] == 1
        (e, n), back = Zygote.pullback(x -> run_wp1(ts, x), B)
    elseif ts.Params["wp"] == 2
        (e, n), back = Zygote.pullback(x -> run_wp2(ts, x), B)
    elseif ts.Params["wp"] == 3
        (e, n), back = Zygote.pullback(x -> run_wp3(ts, x), B)
    elseif ts.Params["wp"] == 4
        (e, n), back = Zygote.pullback(x -> run_wp4(ts, x), B)
    end
    
    gradH = back((1, nothing))[1]
    gradN = back((nothing, 1))[1]

    fprint("wp value: $(e/n)")

    gradH[:], gradN[:]
end

function run_wp(ts, B)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)

    psi1, psi2, psi3, psi4, npsi = get_A3B(ts.A, B)

    wpsi1 = act_wp(psi1)
    wpsi2 = act_wp(psi2)
    wpsi3 = act_wp(psi3)
    wpsi4 = act_wp(psi4)

    w1, n1 = get_wp(n_dm, psi1, wpsi1)
    w2, n2 = get_wp(n_dm, psi2, wpsi2)
    w3, n3 = get_wp(n_dm, psi3, wpsi3)
    w4, n4 = get_wp(n_dm, psi4, wpsi4)

    # nA = get_wp(n_dm, npsi, npsi)

    wp = w1 .+ w2 .+ w3 .+ w4
    # wp = w1 .+ w2
    nB = (n1 .+ n2 .+ n3 .+ n4)/4
    # nB = (n1 .+ n2)/2

    fprint("w1: $(w1/nB)    n1: $(n1)\nw2: $(w2/nB)    n2: $(n2)\nw3: $(w3/nB)    n3: $(n3)\nw4: $(w4/nB)    n4: $(n4)")

    wp, nB
end
function run_wp1(ts, B)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)

    psi1, psi2, psi3, psi4, npsi = get_A3B(ts.A, B)

    wpsi1 = act_wp(psi1)

    w1, n1 = get_wp(n_dm, psi1, wpsi1)

    # nA = get_wp(n_dm, npsi, npsi)
    wp = w1
    nB = n1 

    fprint("w1: $(w1/nB)    n1: $(n1)")

    wp, nB
end
function run_wp2(ts, B)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)

    psi1, psi2, psi3, psi4, npsi = get_A3B(ts.A, B)

    wpsi2 = act_wp(psi2)

    w2, n2 = get_wp(n_dm, psi2, wpsi2)

    # nA = get_wp(n_dm, npsi, npsi)
    wp = w2
    nB = n2 

    fprint("w2: $(w2/nB)    n2: $(n2)")

    wp, nB
end
function run_wp3(ts, B)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)

    psi1, psi2, psi3, psi4, npsi = get_A3B(ts.A, B)

    wpsi3 = act_wp(psi3)

    w3, n3 = get_wp(n_dm, psi3, wpsi3)

    # nA = get_wp(n_dm, npsi, npsi)
    wp = w3
    nB = n3

    fprint("w3: $(w3/nB)    n3: $(n3)")

    wp, nB
end
function run_wp4(ts, B)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)

    psi1, psi2, psi3, psi4, npsi = get_A3B(ts.A, B)

    wpsi4 = act_wp(psi4)

    w4, n4 = get_wp(n_dm, psi4, wpsi4)

    # nA = get_wp(n_dm, npsi, npsi)
    wp = w4
    nB = n4

    fprint("w4: $(w4/nB)    n4: $(n4)")

    wp, nB
end
function run_wp12(ts, B)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)

    psi1, psi2, psi3, psi4, npsi = get_A3B(ts.A, B)

    wpsi1 = act_wp(psi1)
    wpsi2 = act_wp(psi2)

    w1, n1 = get_wp(n_dm, psi1, wpsi1)
    w2, n2 = get_wp(n_dm, psi2, wpsi2)

    # nA = get_wp(n_dm, npsi, npsi)
    wp = w1 .+ w2
    nB = (n1 .+ n2)/2

    fprint("w1: $(w1/nB)    n1: $(n1)\nw2: $(w2/nB)    n2: $(n2)")

    wp, nB
end

function run_wp13()
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)

    psi1, psi2, psi3, psi4, npsi = get_A3B(ts.A, B)

    wpsi1 = act_wp(psi1)
    wpsi3 = act_wp(psi3)

    w1, n1 = get_wp(n_dm, psi1, wpsi1)
    w3, n3 = get_wp(n_dm, psi3, wpsi3)

    # nA = get_wp(n_dm, npsi, npsi)

    wp = w1 .+ w3
    nB = (n1 .+ n3)/2

    fprint("w1: $(w1/nB)    n1: $(n1)\nw3: $(w3/nB)    n3: $(n3)\n")

    wp, nB
end
function run_wp23()
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)

    psi1, psi2, psi3, psi4, npsi = get_A3B(ts.A, B)

    wpsi2 = act_wp(psi2)
    wpsi3 = act_wp(psi3)

    w2, n2 = get_wp(n_dm, psi2, wpsi2)
    w3, n3 = get_wp(n_dm, psi3, wpsi3)

    # nA = get_wp(n_dm, npsi, npsi)

    wp = w2 .+ w3
    nB = (n2 .+ n3)/2

    fprint("w2: $(w2/nB)    n2: $(n2)\nw3: $(w3/nB)    n3: $(n3)\n")

    wp, nB
end
"""
    w1 - w3
    |     |
    w2 - w4
"""
function get_A3B(A, B)
    D = size(A, 1)
    d = size(A, 5)

    psi1 = get_AB(B, A, A, A)
    psi1 = reshape(psi1 , D^8, d, d, d, d)

    psi2 = get_AB(A, B, A, A)
    psi2 = reshape(psi2 , D^8, d, d, d, d)

    psi3 = get_AB(A, A, B, A)
    psi3 = reshape(psi3 , D^8, d, d, d, d)

    psi4 = get_AB(A, A, A, B)
    psi4 = reshape(psi4 , D^8, d, d, d, d)

    npsi = get_AB(A, A, A, A)
    npsi = reshape(npsi, D^8, d, d, d, d)

    return psi1, psi2, psi3, psi4, npsi
end

function get_AB(A, B, C, D)
    AB = tcon([A, B], [[-1,-2, 1,-5,-7], [1,-3,-4,-6,-8]])
    CD = tcon([C, D], [[-1,-2, 1,-5,-7], [1,-3,-4,-6,-8]])

    ABCD = tcon([AB, CD], [[-1,-3,-4,-5,1,2,-9,-10], [-2,1,2,-6,-7,-8,-11,-12]])

    ABCD
end

function act_wp(psi)
    w1 = iPEPS.tout(sI, sigmax)
    w2 = iPEPS.tout(sigmaz, sigmay)
    w3 = iPEPS.tout(sigmay, sigmaz)
    w4 = iPEPS.tout(sigmax, sI)

    @ein wpsi[m1, m2, m3, m4, m5] := psi[m1, p1, p2, p3, p4] * w1[m2, p1] * w2[m3, p2] * w3[m4, p3] * w4[m5, p4];

    wpsi
end

function get_wp(n_dm, psi, wpsi)
    psi = reshape(psi, size(psi,1), :)
    psid = conj(psi)
    wpsi = reshape(wpsi, size(wpsi,1), :)

    wp = tr(transpose(wpsi) * n_dm * psid)

    n = tr(transpose(psi) * n_dm * psid)
    # @show wp, n
    wp, n
end

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