function print_cfg(cfg::Dict)
    keys = ["out_prefix", "model", "dir", "D", "chi", "resume", "es_resume", "es_num","basis", "basis_t", "ad", "gc", "save", "part_save", "basis_cut","nrmB_cut",  "min_iter", "max_iter", "ad_max_iter", "rg_tol", "wp", "wp_name"]
    vals = get.(Ref(cfg), keys, missing)

    for i in eachindex(keys)
        @printf("    %-10s =   %s \n", keys[i], vals[i]) 
    end
end