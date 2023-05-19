function plot_band(n, filename)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load dafault config file")
    end
    print_cfg(cfg)

    px, py = make_es_path()
    E = zeros(n, length(px))
    for i in eachindex(px)
        Ei, vi = evaluate_es(px[i], py[i], cfg)
        E[:,i] = real.(Ei[1:n])
    end

    E
end

function plot_spectral()
    
end