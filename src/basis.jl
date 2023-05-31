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

function wrap_svd(A, n = Inf)
    u,s,v = svd(A)
    ix = sortperm(s)[end:-1:1]
    s = s[ix]
    u = u[:,ix]
    v = v[:,ix]

    u,s,v = cutoff_matrix(u,s,v, 1e-12, n)
    
    u,s, v
end

function cutoff_matrix(u,s,v, cutoff, n)
    n_above_cutoff = count(>(cutoff), s/maximum(s))
    n = min(n, n_above_cutoff) |> Int

    if n < Inf && size(s,1) > n
        if  s[n] > 1e-5 
            n = count(>=(s[n] - cutoff), s)
        end
        u = u[:, 1:n]
        s = s[1:n]
        v = v[:, 1:n]
    end

    u,s,v
end