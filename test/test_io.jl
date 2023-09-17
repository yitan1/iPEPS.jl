using Test
using iPEPS: get_dir, get_gs_name, get_basis_name, get_es_name

@testset "io dir" begin
    params = Dict([("out_prefix", "a"), ("model", "b"), ("D", 2), ("chi", 64)])
    cur_path = pwd()
    dir = get_dir(params)
    # @test "$(cur_path)/simulation/b_a_D2_X64" == dir

    @test "$(dir)/gs.jld2" == get_gs_name(params)
    @test "$(dir)/basis.jld2" == get_basis_name(params)
    @test "$(dir)/es_1_1.jld2" == get_es_name(params, 1.0, 1.0)
    @test "$(dir)/es_1.2_0.1.jld2" == get_es_name(params, 1.2, 0.1)
end