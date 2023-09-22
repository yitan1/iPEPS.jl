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
    # TODO
    # px, py = string(px), string(py)
    if params["part_save"] 
        es_dir = "$(dir)/es_$(px)_$(py)"
        if ! ispath(es_dir)
            println("$es_dir does not exist, and will be created")
            mkpath(es_dir)
        end
        st = params["es_resume"]
        ed = st + params["es_num"] - 1
        es_name = "$es_dir/$(st)_$(ed).jld2"
    else
        es_name = "$(dir)/es_$(px)_$(py).jld2"
    end

    es_name
end

function get_dir(params)
    if params["dir"] == 0
        pre = get(params, "out_prefix", "none")
        model = get(params, "model", "none")
        D = get(params, "D", "none")
        X = get(params, "chi", "none")
        cur_path = pwd()
        dir = "$(cur_path)/simulation/$(model)_$(pre)_D$(D)_X$(X)"
    else 
        dir = params["dir"]
    end

    if ! ispath(dir)
        println("$dir does not exist, and will be created")
        mkpath(dir)
    end

    dir
end