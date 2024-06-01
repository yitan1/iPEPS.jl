struct Config
    out_prefix::String
    model::String
    dir::String
    D::Int
    chi::Int
    resume::Bool
    es_resume::Bool
    es_num::Int
    basis::String
end

struct GsConfig
    out_prefix::String
    model::String
    dir::String
    D::Int
    chi::Int
    resume::Bool

    function GsConfig(; out_prefix::String="default", model::String="ising", dir::String="0", D::Int=2, chi::Int=32, resume::Bool=true)
        new(out_prefix, model, dir, D, chi, resume)
    end
end

struct EsConfig
    out_prefix::String
    model::String
    dir::String
    D::Int
    chi::Int
    resume::Bool
    es_resume::Bool
    es_num::Int
    basis::String
    basis_t::String
    ad::Bool
    gc::Bool
    save::Bool
    part_save::Bool
    basis_cut::Int
    nrmB_cut::Float64
    min_iter::Int
    max_iter::Int
    ad_max_iter::Int
    rg_tol::Float64
    wp::Bool
    wp_name::String

    function EsConfig(; out_prefix::String="default", model::String="ising", dir::String="0",
        D::Int=2, chi::Int=32,
        es_resume::Bool=true, es_num::Int=1,
        basis::String="default", basis_t::String="default",
        ad::Bool=true, gc::Bool=true, save::Bool=true,
        part_save::Bool=true, basis_cut::Int=1, nrmB_cut::Float64=1e-12,
        min_iter::Int=1, max_iter::Int=200, ad_max_iter::Int=200, rg_tol::Float64=1e-12)
        new(out_prefix, model, dir, D, chi, resume, es_resume, es_num, basis, basis_t, ad, gc, save, part_save, basis_cut, nrmB_cut, min_iter, max_iter, ad_max_iter, rg_tol, wp, wp_name)
    end
end

struct WpConfig

end

function get_default_config()
    cfg = TOML.parsefile("src/optimize/default_config.toml")
    return cfg
end

function print_cfg(cfg::Dict)
    keys = ["out_prefix", "model", "dir", "D", "chi", "resume", "es_resume", "es_num", "basis", "basis_t", "ad", "gc", "save", "part_save", "basis_cut", "nrmB_cut", "min_iter", "max_iter", "ad_max_iter", "rg_tol", "wp", "wp_name"]
    vals = get.(Ref(cfg), keys, missing)

    for i in eachindex(keys)
        @printf("    %-10s =   %s \n", keys[i], vals[i])
    end
end