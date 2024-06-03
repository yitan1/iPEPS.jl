function plot_band(n, filename)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/optimize/default_config.toml")
        fprint("load dafault config file")
    end

    plot_band(n, cfg)
end

function plot_band(n, cfg::Dict)   
    print_cfg(cfg)
    px, py = make_es_path()
    E = zeros(n, length(px))
    for i in eachindex(px)
        Ei, vi = evaluate_es(px[i], py[i], cfg)
        E[:,i] = real.(Ei[1:n])
    end

    E
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