# @Zygote.nograd time
ChainRulesCore.@non_differentiable time(::Any...)
ChainRulesCore.@non_differentiable Printf.format(::Any...)
ChainRulesCore.@non_differentiable append!(::Any...)
# ChainRulesCore.@non_differentiable maximum(::Any...)
ChainRulesCore.@non_differentiable get(::Any...)

###### Copy from the pkg of BackwardsLinalg 
struct ZeroAdder end

Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

mpow2(a::AbstractArray) = a .^ 2

"""
		svd(A) -> Tuple{AbstractMatrix, AbstractVector, AbstractMatrix}
"""
function svd1(A)
    U, S, V = LinearAlgebra.svd(A)
    return U, S, Matrix(V)
end

Zygote.@adjoint function svd1(A)
    U, S, V = svd1(A)
    (U, S, V), dy -> (svd_back(U, S, V, dy...),)
end

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-12) where {T}
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = mpow2(S)
    Sinv = @. S / (S2 + η)
    F = S2' .- S2
    F ./= (mpow2(F) .+ η)

    res = ZeroAdder()
    if !(dU isa Nothing)
        UdU = U' * dU
        J = F .* (UdU)
        res += (J + J') * LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im * imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa Nothing)
        VdV = V' * dV
        K = F .* (VdV)
        res += LinearAlgebra.Diagonal(S) * (K + K')
    end
    if !(dS isa Nothing)
        res += LinearAlgebra.Diagonal(dS)
    end

    res = U * res * V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U * (U' * dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV' * V) * V')
    end
    res
end