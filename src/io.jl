function get_gs_name(params)
    dir = get_dir(params)
    gs_name = "$(dir)/gs.jld2"
    
    gs_name
end

function get_basis_name(params)
    dir = get_dir(params)
    basis_name = "$(dir)/basis.jld2"

    basis_name
end

function get_es_name(params, px, py)
    dir = get_dir(params)
    es_name = "$(dir)/es_$(px)_$(py).jld2"
    
    es_name
end

function get_dir(params)
    pre = get(params, "out_prefix", "none")
    model = get(params, "model", "none")
    D = get(params, "D", "none")
    X = get(params, "chi", "none")
    dir = "simulation/$(model)_$(pre)_D$(D)_X$(X)"
    
    if ! ispath(dir)
        mkdir(dir)
    end

    dir
end