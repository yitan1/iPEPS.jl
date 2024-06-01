function get_gs_name(params)
    dir = get_dir(params)
    gs_name = "$(dir)/gs.jld2"

    gs_name
end

function get_basis_name(params)
    dir = get_dir(params)
    if params["basis_name"] == 0
        basis_name = "$(dir)/basis.jld2"
    else
        basis_name = params["basis_name"]
        basis_name = "$(dir)/$basis_name.jld2"
    end

    basis_name
end

function get_es_name(params, px, py)
    dir = get_dir(params)
    # TODO
    # px, py = string(px), string(py)
    if params["part_save"]
        es_dir = "$(dir)/es_$(px)_$(py)"
        if !ispath(es_dir)
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

function get_wp_name(params)
    dir = get_dir(params)
    wp_name = params["basis_name"]

    if !occursin("wp/", wp_name)
        dir = "$(dir)/wp"
        if !ispath(dir)
            println("$dir does not exist, and will be created")
            mkpath(dir)
        end
    end

    if occursin("basis", wp_name)
        wp_name = replace(wp_name, "basis" => "HN")
    end

    if params["part_save"]
        wp_dir = "$(dir)/$(wp_name)"
        if !ispath(wp_dir)
            println("$wp_dir does not exist, and will be created")
            mkpath(wp_dir)
        end
        st = params["es_resume"]
        ed = st + params["es_num"] - 1
        wp_name1 = "$wp_dir/$(st)_$(ed).jld2"
    else
        wp_name1 = "$(dir)/$(wp_name).jld2"
    end

    wp_name1
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

    if !ispath(dir)
        println("$dir does not exist, and will be created")
        mkpath(dir)
    end

    dir
end