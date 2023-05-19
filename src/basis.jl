import LinearAlgebra: norm, tr

@Zygote.nograd time
@Zygote.nograd Printf.format
@Zygote.nograd append!
@Zygote.nograd maximum
@Zygote.nograd get

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