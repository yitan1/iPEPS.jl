using Zygote
using LinearAlgebra

@Zygote.nograd get_projector

# @Zygote.adjoint function IPEPS{LT,T,N,AT}(bulk) where {LT,T,N,AT}
#     IPEPS{LT,T,N,AT}(bulk), dy -> (dy.bulk,)
# end

@Zygote.adjoint function EnvTensor(bulk::AT, corner::Vector{CT}, edge::Vector{ET}, chi) where {T, N,
                    AT<:AbstractArray{T,N}, CT<:AbstractArray{T}, ET<:AbstractArray{T}}
        return EnvTensor{T, N, AT, CT, ET}(bulk, corner, edge, chi), dy->(dy.bulk, dy.corner, dy.edge, dy.chi)
end


@Zygote.adjoint function LinearAlgebra.norm(A::AbstractArray, p::Real = 2)
    n = norm(A,p)
    back(Δ) = let n = n
                    (Δ .* A ./ (n + eps(0f0)),)
                end
    return n, back
end