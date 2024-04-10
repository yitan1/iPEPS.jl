function plot_band(n, filename)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/optimize/default_config.toml")
        fprint("load dafault config file")
    end
    print_cfg(cfg)
    plot_band(n, cfg)
end

function plot_band(n, cfg::Dict)
    px, py = make_es_path()
    E = zeros(n, length(px))
    for i in eachindex(px)
        Ei, vi = evaluate_es(px[i], py[i], cfg)
        E[:,i] = real.(Ei[1:n])
    end

    E
end

function basis_dep(n, px, py, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    # print_cfg(cfg)

    es_name = get_es_name(cfg, px, py)
    effH = load(es_name, "effH")
    effN = load(es_name, "effN")
    fprint("load H and N at $es_name")
    H = (effH + effH') /2 
    N = (effN + effN') /2
    ev_N, P = eigen(N)
    idx = sortperm(real.(ev_N))[end:-1:1]
    ev_N = ev_N[idx]
    # display(ev_N/maximum(ev_N))
    selected = ev_N .> ev_N[n+1]
    # display(ev_N[selected] /maximum(ev_N))
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

    es, vecs
end

function plot_spectral(es, swk0; step = 0.1, factor = 0.04, x_max = 0)
    if x_max > 0
        x_max = x_max
    else
        x_max = ceil(maximum(es))
    end
    x = 0:step:x_max

    y =  [lor_broad(x[i], es, swk0, factor) for i in eachindex(x)]

    x, y
end