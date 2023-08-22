function compute_es(px, py, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)

    es_name = get_es_name(cfg, px, py)
    effH = load(es_name, "effH")
    effN = load(es_name, "effN")
    fprint("load H and N at $es_name")

    nrmB_cut = get(cfg, "nrmB_cut", 1e-3)

    H = (effH + effH') /2 
    N = (effN + effN') /2
    ev_N, P = eigen(N)
    idx = sortperm(real.(ev_N))[end:-1:1]
    ev_N = ev_N[idx]
    display(ev_N/maximum(ev_N))
    if nrmB_cut isa Int
        selected = ev_N .> ev_N[nrmB_cut+1]
    else
        selected = (ev_N/maximum(ev_N) ) .> nrmB_cut
    end
    display(ev_N[selected] /maximum(ev_N))
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

function compute_spec_env(op, px, py, filename::String)
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

    ts.Params["px"] = convert(eltype(ts.A), px*pi)
    ts.Params["py"] = convert(eltype(ts.A), py*pi)

    B = tcon([ts.A, op], [[-1,-2,-3,-4,1], [-5,1]])
    # Bd = conj(B)

    Cs, Es = init_ctm(ts.A, ts.Ad)

    ts = setproperties(ts, Cs = Cs, Es = Es, B = B, Bd = conj(B))

    # conv_fun(_x) = get_es_energy(_x, H)
    ts, _ = run_ctm(ts)

    _, envB = get_all_norm(ts)

    envB, basis
end

function lor_broad(x, es, swk, factor)
    w = 0.0
    for i in eachindex(swk)
        w += 1/pi*factor/((x - es[i])^2 + factor^2)*swk[i]
    end

    w
end