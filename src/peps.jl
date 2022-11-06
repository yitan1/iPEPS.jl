abstract type PEPS end
"""
    IPEPS

dim(A): d*D*D*D*D (phy, up, left, down, right)
"""
struct IPEPS{T, N, AT<:AbstractArray{T,N}} <: PEPS
    A::AT
end

IPEPS(A::AbstractArray{T,N}) where {T,N} = IPEPS{T, N, typeof(A)}(A) 

struct ExcIPEPS{T, N, AT<:AbstractArray{T,N}} <: PEPS
    kx::T
    ky::T
    A::AT
    B::AT
end

ExcIPEPS(kx, ky, A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} = IPEPS{T, N, typeof(A)}(kx, ky, A, B) 


transfer_matrix(A::AbstractArray) = transfer_matrix(A, conj(A)) 

function transfer_matrix(A::AbstractArray,B::AbstractArray)
    @tensor T[a1,a2,b1,b2,c1,c2,d1,d2] := A[i, a1, b1, c1, d1]*B[i, a2, b2, c2, d2]
    dim = size(T)
    T = reshape(T, dim[1]*dim[2], dim[3]*dim[4], dim[5]*dim[6], dim[7]*dim[8])
    T
end