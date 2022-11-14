abstract type PEPS end
"""
    IPEPS

- dim(A): dD⁴ (phy, up, left, down, right)
    ```
         1
         |
    2 -- A -- 4
         |
         3
    ```
"""
struct IPEPS{T, N, AT<:AbstractArray{T,N}} <: PEPS
    A::AT
end

IPEPS(A::AbstractArray{T,N}) where {T,N} = IPEPS{T, N, typeof(A)}(A) 

data(phi::IPEPS) = phi.A

struct ExcIPEPS{T, N, AT<:AbstractArray{T,N}} <: PEPS
    kx::T
    ky::T
    A::AT
    B::AT
end

ExcIPEPS(kx, ky, A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N} = IPEPS{T, N, typeof(A)}(kx, ky, A, B) 


