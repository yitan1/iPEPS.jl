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
#TODO
function correlation_spin(op, cfg)
    bs_name = get_basis_name(cfg)
    ts0 = load(bs_name, "ts")
    ts = get_spin_boundary(op, ts0)

    ts.Params["max_iter"] = 1
    max_iter = 10
    for i = 1:max_iter
        ts, _ = run_ctm(ts)
        ndm_Ad = get_env_A(ts)
        # compute A * S * n_dm_Ad

        println("correlation: ", nrm0[1])
    end
end

function get_spin_boundary(op, ts0)
    A_op = tcon([ts0.A, op], [[-1, -2, -3, -4, 1], [1, -5]])

    chi = ts0.Params["chi"]

    ts = setproperties(ts0, A=A_op)
    ts, _ = left_rg(ts, chi)
    ts = setproperties
    ts
end

function get_env_A(ts)
    # A = ts.A
    Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es
    n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)
    ndm_Ad = tcon([n_dm, Ad], [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5]])
    # nrm0 = tcon([ndm_Ad, A], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

    return ndm_Ad
end