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

IPEPS(A::AbstractArray) = IPEPS(A)

IPEPS(A::AbstractArray{T,N}) where {T,N} = IPEPS{T, N, typeof(A)}(A) 

get_A(phi::IPEPS) = phi.A
get_Ad(phi::IPEPS) = conj(phi.A)

struct ExcIPEPS{T, N, AT<:AbstractArray{T,N}} <: PEPS
    kx::T
    ky::T
    GS::IPEPS
    B::AT
end

ExcIPEPS(kx, ky, GS::IPEPS, B::AbstractArray{T,N}) where {T,N} = IPEPS{T, N, typeof(B)}(kx, ky, GS, B) 

get_GS(ES::ExcIPEPS) = ES.GS
get_A(ES::ExcIPEPS) = get_GS(ES) |> get_A
get_Ad(ES::ExcIPEPS) = get_GS(ES) |> get_Ad
get_B(ES::ExcIPEPS) = ES.B
get_Bd(ES::ExcIPEPS) = conj(ES.B)


