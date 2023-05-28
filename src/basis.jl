import LinearAlgebra: norm, tr

# @Zygote.nograd time
ChainRulesCore.@non_differentiable time(::Any...)
ChainRulesCore.@non_differentiable Printf.format(::Any...)
ChainRulesCore.@non_differentiable append!(::Any...)
# ChainRulesCore.@non_differentiable maximum(::Any...)
ChainRulesCore.@non_differentiable get(::Any...)

# nograd(x) = x
# @Zygote.nograd nograd

# @Zygote.adjoint function LinearAlgebra.norm(A::AbstractArray,p::Real = 2)
#     n = norm(A,p)
#     back(Δ) = let n = n
#                     (Δ .* A ./ (n + eps(0f0)),)
#                 end
#     return n, back
# end

renormalize(A::AbstractArray) = A ./ maximum(abs.(A))
# renormalize(A::AbstractArray) = A ./ norm(A)


function diag_inv(A::AbstractArray)
    diagm(1 ./ A)
end