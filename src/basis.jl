import LinearAlgebra: norm
using Printf

Base.print(io::IO, x::Float64) = @printf(io, "%.5g", x)
# function Base.print(io::IO, x::Float64) 
#     if abs(x) < 1e-4
#         return  @printf(io, "%0.6g", x)
#     else 
#         return @printf(io, "%0.4f", x)
#     end
# end
Base.print(io::IO, x::ComplexF64) = @printf(io, "%0.6f + %0.6f", real(x), imag(x))

@Zygote.nograd time
@Zygote.nograd Printf.format
@Zygote.nograd append!
@Zygote.nograd maximum

nograd(x) = x
@Zygote.nograd nograd

# @Zygote.adjoint function LinearAlgebra.norm(A::AbstractArray,p::Real = 2)
#     n = norm(A,p)
#     back(Δ) = let n = n
#                     (Δ .* A ./ (n + eps(0f0)),)
#                 end
#     return n, back
# end

renormalize(A::AbstractArray) = A ./ maximum(abs, A)
# renormalize(A::AbstractArray) = A ./ norm(A)


function diag_inv(A::AbstractArray)
    diagm(1 ./ A)
end

# @Zygote.adjoint function LinearAlgebra.norm(A::AbstractArray, p::Real = 2)
#     n = norm(A,p)
#     back(Δ) = let n = n
#                     (Δ .* A ./ (n + eps(0f0)),)
#                 end
#     return n, back
# end