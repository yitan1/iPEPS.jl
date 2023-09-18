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





