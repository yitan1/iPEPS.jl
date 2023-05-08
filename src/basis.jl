using Printf
# Base.show(io::IO, x::Float64) = @printf(io, "%0.4e", x)
Base.print(io::IO, x::Float64) = @printf(io, "%0.4e", x)
Base.print(io::IO, x::ComplexF64) = @printf(io, "%0.4e + %0.4e", real(x), imag(x))

@Zygote.nograd time
@Zygote.nograd append!
nograd(x) = x
@Zygote.nograd nograd
# @Zygote.nograd CTMTensors

renormalize(A::AbstractArray) = A ./ maximum(abs, A)


