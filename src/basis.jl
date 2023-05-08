using Printf
# Base.show(io::IO, x::Float64) = @printf(io, "%0.4e", x)
function Base.print(io::IO, x::Float64) 
    if abs(x) < 1e-4
        return  @printf(io, "%0.4e", x)
    else 
        return @printf(io, "%0.4f", x)
    end
end
Base.print(io::IO, x::ComplexF64) = @printf(io, "%0.4f + %0.4f", real(x), imag(x))

@Zygote.nograd time
@Zygote.nograd append!
nograd(x) = x
@Zygote.nograd nograd
# @Zygote.nograd CTMTensors

renormalize(A::AbstractArray) = A ./ maximum(abs, A)


