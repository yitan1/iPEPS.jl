using iPEPS
using Test

@testset "iPEPS.jl" begin
    @testset "print test" begin
        include("test_print.jl")
    end

    @testset "io test" begin
        include("test_io.jl")
    end

    @testset "tcon test" begin
        include("test_tcon.jl")
    end

    @testset "model test" begin
        include("test_model.jl")
    end
end


using iPEPS
using JLD2, TOML

A = load("As4_JK.jld2")["As"]

cfg = TOML.parsefile("src/optimize/default_config.toml")
ts = iPEPS.CTMTensors(A, cfg);
ts, _ = iPEPS.run_ctm(ts, conv_fun = nothing);
iPEPS.run_wp(ts, A, A, A, A)



