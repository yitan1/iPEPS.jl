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

ExcIPEPS(A::AbstractArray, B::AbstractArray) = ExcIPEPS(IPEPS(A), B)
ExcIPEPS(GS::IPEPS, B::AbstractArray) = ExcIPEPS(0.0, 0.0, GS, B)
ExcIPEPS(kx, ky, GS::IPEPS, B::AbstractArray{T,N}) where {T,N} = IPEPS{T, N, typeof(B)}(kx, ky, GS, B) 

get_kx(ES::ExcIPEPS) = ES.kx
get_ky(ES::ExcIPEPS) = ES.ky
get_GS(ES::ExcIPEPS) = ES.GS
get_B(ES::ExcIPEPS) = ES.B
get_Bd(ES::ExcIPEPS) = conj(ES.B)

get_A(ES::ExcIPEPS) = get_GS(ES) |> get_A
get_Ad(ES::ExcIPEPS) = get_GS(ES) |> get_Ad


