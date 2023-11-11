function fprint(xs)
    println(xs)
    flush(stdout)
end
Base.print(io::IO, x::Float64) = @printf(io, "%.5g", x)
# function Base.print(io::IO, x::Float64) 
#     if abs(x) < 1e-4
#         return  @printf(io, "%0.6g", x)
#     else 
#         return @printf(io, "%0.4f", x)
#     end
# end
Base.print(io::IO, x::ComplexF64) = @printf(io, "%0.6f %0.6f im", real(x), imag(x))

function print_cfg(cfg::Dict)
    keys = ["out_prefix", "model", "dir", "D", "chi", "resume", "es_resume", "es_num","basis", "basis_t", "ad", "gc", "save", "part_save", "basis_cut","nrmB_cut",  "min_iter", "max_iter", "ad_max_iter", "rg_tol", "wp", "wp_name"]
    vals = get.(Ref(cfg), keys, missing)

    for i in eachindex(keys)
        @printf("    %-10s =   %s \n", keys[i], vals[i]) 
    end
end