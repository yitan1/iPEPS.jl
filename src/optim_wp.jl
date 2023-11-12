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
    if  ts.Params["es_resume"] > 0 && ispath(wp_name) 
        wp_file = load(wp_name)
        effH = wp_file["effH"]
        effN = wp_file["effN"]
        fprint("load existed calculation , effH, effN in $wp_name")
    else
        effH = zeros(ComplexF64, basis_dim, basis_dim)
        effN = zeros(ComplexF64, basis_dim, basis_dim)
    end

    for i = 1:basis_dim
        if i < ts.Params["es_resume"]
            fprint(" Simulation of basis vector $(i)/$(basis_dim) existed, skip to next")
            continue
        end
        if ts.Params["es_num"] > 0 && i >= (ts.Params["es_resume"] + ts.Params["es_num"])
            fprint("\nUp to maximum simulation of basis vector $(i)/$(basis_dim) existed, end to calculation")
            break
        end
        fprint("\nStarting simulation of basis vector $(i)/$(basis_dim)")

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
    A = ts.A

    # (e, n), back = Zygote.pullback(x -> run_wp_exact(ts, x), B)
    if ts.Params["wp"] == 1
        f = x -> run_wp(ts, x, A, A, A)
    elseif ts.Params["wp"] == 2
        f = x -> run_wp(ts, A, x, A, A)
    elseif ts.Params["wp"] == 3
        f = x -> run_wp(ts, A, A, x, A)
    elseif ts.Params["wp"] == 4
        f = x -> run_wp(ts, A, A, A, x)
    end

    (e, n), back = Zygote.pullback(f, B)
    gradH = back((1, nothing))[1]
    gradN = back((nothing, 1))[1]

    fprint("wp value: $(e/n)")

    gradH[:], gradN[:]
end

function run_wp(ts, B1, B2, B3, B4)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4, B1, B2, B3, B4)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:4]), :)

    w_op = get_w_op()
    wp = tr(w_op*n_dm)
    nB = tr(n_dm)

    fprint("wp: $(wp/nB)    nB: $(nB)")

    wp, nB
end

function run_wp_all(ts, B)
    A = ts.A
    w1, n1 = run_wp(ts, B, A, A, A)
    w2, n2 = run_wp(ts, A, B, A, A)
    w3, n3 = run_wp(ts, A, A, B, A)
    w4, n4 = run_wp(ts, A, A, A, B)

    fprint("w1: $(w1/n1)    n1: $(n1)\nw2: $(w2/n2)    n2: $(n2)\nw3: $(w3/n3)    n3: $(n3)\nw4: $(w4/n4)    n4: $(n4)")

    w1, w2, w3, w4, n1, n2, n3, n4
end

function run_wp12(ts, B)
end
function run_wp13()
end
function run_wp23()
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
        C1E1 = tcon([C1, E1], [[-1,1], [1,-2,-3,-4]])
        CEE4 = tcon([C1E1, E4], [[1,-2,-3,-5], [1,-1,-4,-6]])
        CEEB = tcon([CEE4, B1], [[-1,-2,1,2,-3,-4], [1,2,-5,-6,-7]])
        tcon([CEEB, conj(B1)], [[-1,-2, 1,2, -3,-4,-7], [1,2,-5,-6,-8]])
    end

    block2 = begin
        C4E3 = tcon([C4, E3], [[-1,1], [1,-2,-3,-4]])
        CEE4 = tcon([C4E3, E4], [[1,-2,-4,-6],[-1,1,-3,-5]])
        CEEB = tcon([CEE4, B2], [[-1,-2,1,2,-3,-4], [-5,1, 2,-6,-7]])
        tcon([CEEB, conj(B2)], [[-1,-2,1,2,-3,-4,-7], [-5,1, 2,-6,-8]])
    end
    block_L = tcon([block1, block2], [[1,-1,2,-3,3,-5,-7,-9], [1,-2,2,-4,3,-6,-8,-10]])

    block3 = begin
        C2E1 = tcon([C2, E1], [[1,-2], [-1, 1, -3, -4]])
        CEE2 = tcon([C2E1, E2], [[-1,1,-3,-5], [1,-2,-4,-6]])
        CEEB = tcon([CEE2, B3], [[-1,-2,1,2,-3,-4], [1,-5,-6, 2,-7]])
        tcon([CEEB, conj(B3)], [[-1,-2,1,2,-3,-4,-7], [1,-5,-6,2,-8]])
    end

    block4 = begin
        C3E3 = tcon([C3, E3], [[-1,1], [-2,1,-3,-4]])
        CEE2 = tcon([C3E3, E2], [[1,-2,-3,-5], [-1,1,-4,-6]])
        CEEB = tcon([CEE2, B4], [[-1,-2,1,2,-3,-4], [-5,-6, 1,2,-7]])
        tcon([CEEB, conj(B4)], [[-1,-2,1,2,-3,-4,-7], [-5,-6, 1,2,-8]])
    end

    block_R = tcon([block3, block4], [[-1,1,-3,2,-5,3,-7,-9], [1,-2,2,-4,3,-6,-8,-10]])

    dm4 = tcon([block_L, block_R], [[1,2,3,4,5,6,-1,-2,-5,-6], [1,2,3,4,5,6,-3,-4,-7,-8]])

    dm4
end
"""
    w1 - w3
    |     |
    w2 - w4
"""
function get_w_op()
    w1 = tout(sI, sigmax)
    w2 = tout(sigmaz, sigmay)
    w3 = tout(sigmay, sigmaz)
    w4 = tout(sigmax, sI)

    wp = tout(tout(tout(w1, w2), w3), w4)

    wp
end

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