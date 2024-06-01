"""
1. get effH, effN of wp1 for full basis
2. diag effH, effN and get new basis wp_1p, wp_1m
3. get effH, effN of wp2 for basis wp_1p, wp_1m
4. diag effH, effN and get new basis for wp_1p2p, wp_1p2m, wp_1m2p, wp_1m2m 
.......

get new 2^4 basis basis_xxxx 
"""
function get_wp_basis(cfg)
    dir = get_dir(cfg)
    basis_dir = get_basis_name(cfg)

    basis_name = cfg["basis_name"]
    if !occursin("wp/", basis_name)
        basis_name = "wp/$(basis_name)"
    end

    HN_dir = get_wp_name(cfg)
    wp = cfg["wp"]
    new_basis_name = ["$(dir)/$(basis_name)_$(wp)p.jld2", "$(dir)/$(basis_name)_$(wp)m.jld2"]

    get_wp_basis(HN_dir, basis_dir, new_basis_name)
end
function get_wp_basis(HN_dir, basis_dir, new_basis_dir)
    dir = HN_dir
    if isdir(dir)
        list = readdir(dir, join=true, sort=true)
        shape = load(list[1], "effH") |> size
        effH = zeros(ComplexF64, shape)
        effN = zeros(ComplexF64, shape)
        for name in list
            effH .+= load(name, "effH")
            effN .+= load(name, "effN")
        end
    else 
        effH = load(dir, "effH")
        effN = load(dir, "effN")
    end

    H = (effH + effH') / 2
    N = (effN + effN') / 2
    ev_N, P = eigen(N)
    idx = sortperm(real.(ev_N))[end:-1:1]
    ev_N = ev_N[idx]
    P = P[:, idx]
    N2 = P' * N * P
    H2 = P' * H * P
    H2 = (H2 + H2') / 2
    N2 = (N2 + N2') / 2
    es, vecs = eigen(H2, N2)
    ixs = sortperm(real.(es))
    es = es[ixs]
    vecs = vecs[:, ixs]

    display(es)

    f = load(basis_dir)
    basis = f["basis"]
    exci_n = basis * P * vecs

    selected = real.(es) .> 0.0
    nbasisp = exci_n[:, selected]

    selected = real.(es) .< 0.0
    nbasism = exci_n[:, selected]

    jldsave(new_basis_dir[1], basis=nbasisp, ts=f["ts"], H=f["H"])
    jldsave(new_basis_dir[2], basis=nbasism, ts=f["ts"], H=f["H"])
end

function evaluate_wp(filename::String; disp=true)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load dafault config file")
    end

    evaluate_wp(cfg; disp=disp)
end
function evaluate_wp(cfg::Dict; disp=true)
    print_cfg(cfg)

    wp_name = get_wp_name(cfg)
    effH = load(wp_name, "effH")
    effN = load(wp_name, "effN")
    fprint("load H and N at $wp_name")

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

    es, vecs, P
end

function optim_wp(w_op, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end

    optim_wp(w_op, cfg)
end

function optim_wp(wp_op, cfg::Dict)
    print_cfg(cfg)

    basis_name = get_basis_name(cfg)
    basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")
    ts = setproperties(ts, Params=cfg)
    fprint("load basis, ts in $basis_name")

    basis = complex(basis) # ！！！！ convert Complex
    basis_dim = size(basis, 2)

    wp_name = get_wp_name(ts.Params)
    if ts.Params["es_resume"] > 0 && ispath(wp_name)
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

        @time gH, gN = get_wp_grad(ts, wp_op, basis[:, i])
        effH[:, i] = transpose(conj(basis)) * gH / 2
        effN[:, i] = transpose(conj(basis)) * gN / 2

        fprint("\nFinish basis vector of $(i)/$(basis_dim)")

        if ts.Params["save"]
            jldsave(wp_name; effH=effH, effN=effN)
            fprint("Saved (effH, effN) and envB to $(wp_name)")
        end

        if ts.Params["gc"]
            GC.gc()
        end
    end

    if ts.Params["save"]
        jldsave(wp_name; effH=effH, effN=effN)
        fprint("Saved (effH, effN) and envB to $(wp_name)")
    end

    if ts.Params["gc"]
        GC.gc()
    end

    effH, effN
end

function get_wp_grad(ts, wp_op, Bi)
    B = reshape(Bi, size(ts.A))
    A = ts.A

    wp_op_s4 = get_local_h(wp_op)

    # (e, n), back = Zygote.pullback(x -> run_wp_exact(ts, x), B)
    if ts.Params["wp"] == 1
        f = x -> run_wp(ts, wp_op_s4, x, A, A, A)
    elseif ts.Params["wp"] == 2
        f = x -> run_wp(ts, wp_op_s4, A, x, A, A)
    elseif ts.Params["wp"] == 3
        f = x -> run_wp(ts, wp_op_s4, A, A, x, A)
    elseif ts.Params["wp"] == 4
        f = x -> run_wp(ts, wp_op_s4, A, A, A, x)
    end

    (e, n), back = Zygote.pullback(f, B)
    gradH = back((1, nothing))[1]
    gradN = back((nothing, 1))[1]
    # Nb, gradN = get_all_norm(ts1)

    fprint("wp value: $(e/n)")

    gradH[:], gradN[:]
end

# local wp
function run_wp(ts, wp, A1, A2, A3, A4)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    s1, s2, s3, s4 = wp
    A1d, A2d, A3d, A4d = conj(A1), conj(A2), conj(A3), conj(A4)
    op_A1d, op_A2d, op_A3d, op_A4d = get_op_Ad4(s1, s2, s3, s4, A1d, A2d, A3d, A4d)

    wp, norm = energy_norm_4x4(C1, C2, C3, C4, E1, E2, E3, E4, A1, A2, A3, A4, op_A1d, op_A2d, op_A3d, op_A4d)
    nB = norm[]
    wp_exp = wp[]

    fprint("wp: $(wp_exp/nB) nB: $(norm[])")

    wp_exp, nB
end

# function run_wp(ts, w_op, B1, B2, B3, B4)
#     C1, C2, C3, C4 = ts.Cs
#     E1, E2, E3, E4 = ts.Es

#     n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4, B1, B2, B3, B4)
#     n_dm = reshape(n_dm, prod(size(n_dm)[1:4]), :)

#     w_op = get_w_op()
#     wp = tr(w_op*n_dm)
#     nB = tr(n_dm)

#     fprint("wp: $(wp/nB)    nB: $(nB)")

#     wp, nB
# end

function run_wp_all(ts, wp, B)
    wp = get_local_h(wp)

    A = ts.A
    w1, n1 = run_wp(ts, wp, B, A, A, A)
    w2, n2 = run_wp(ts, wp, A, B, A, A)
    w3, n3 = run_wp(ts, wp, A, A, B, A)
    w4, n4 = run_wp(ts, wp, A, A, A, B)

    fprint("w1: $(w1/n1)    n1: $(n1)\nw2: $(w2/n2)    n2: $(n2)\nw3: $(w3/n3)    n3: $(n3)\nw4: $(w4/n4)    n4: $(n4)")

    wp = w1 + w2 + w3 + w4
    nB = (n1 + n2 + n3 + n4) / 4

    wp, nB
end

# function get_w_op()
#     w1 = tout(sI, sigmax)
#     w2 = tout(sigmaz, sigmay)
#     w3 = tout(sigmay, sigmaz)
#     w4 = tout(sigmax, sI)

#     wp = tout(tout(tout(w1, w2), w3), w4)

#     wp
# end