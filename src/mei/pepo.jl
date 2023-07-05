using OMEinsum, TensorOperations, LinearAlgebra
const σx=Float64[0 1;1 0]
const σy=ComplexF64[0 -1im; 1im 0]
const σz=Float64[1 0;0 -1]
const id2=Float64[1 0;0 1]
const σp=Float64[0 2;0 0]/sqrt(2.0)
const σm=Float64[0 0;2 0]/sqrt(2.0)

abstract type PEPO end

struct PEPOTensors{T,CT<:AbstractArray{T,4},ET<:AbstractArray{T,5},BT<:AbstractArray{T,6}} <: PEPO
    C1::CT
    C2::CT
    C3::CT
    C4::CT
    T1::ET
    T2::ET
    T3::ET
    T4::ET
    A ::BT
end

function init_pepoN()
    #mpo for vertical bonds
    A=zeros(Float64,1,1,1,1,2,2)
    A[1,1,1,1,:,:]=id2;

    C1=ein"jikl->ijkl"(A[1,:,:,1,:,:])
    C2=ein"jikl->ijkl"(A[1,1,:,:,:,:])
    C3=ein"ijkl->ijkl"(A[:,1,1,:,:,:])
    C4=ein"jikl->ijkl"(A[:,:,1,1,:,:])

    T1=ein"jkilm->ijklm"(A[1,:,:,:,:,:])
    T2=ein"ijklm->ijklm"(A[:,1,:,:,:,:])
    T3=ein"kijlm->ijklm"(A[:,:,1,:,:,:])
    T4=ein"jkilm->ijklm"(A[:,:,:,1,:,:])

    return C1,C2,C3,C4,T1,T2,T3,T4,A
end

function init_pepo(τ)
    AL,Ah,AR=exptH(τ)
    AU,Av,AD=exptH(τ)

    C1=ein"iαγ,jγβ->ijαβ"(AL,AU)
    C2=ein"jαγ,iγβ->ijαβ"(AL,AD)
    C3=ein"iαγ,jγβ->ijαβ"(AR,AD)
    C4=ein"jαγ,iγβ->ijαβ"(AR,AU)

    T1=ein"kαγ,ijγβ->ijkαβ"(AL,Av)
    T2=ein"ijαγ,kγβ->ijkαβ"(Ah,AD)
    T3=ein"kαγ,jiγβ->ijkαβ"(AR,Av)
    T4=ein"jiαγ,kγβ->ijkαβ"(Ah,AU)
    
    A=ein"ikαγ,ljγβ->ijklαβ"(Ah,Av)
    return C1,C2,C3,C4,T1,T2,T3,T4,A
end

function exptH(τ)
    Jx,Jy,Jz=1.0, 1.0, 1.0
    h=Jz*ein"ij,kl->ikjl"(σz,σz)-
      Jx*ein"ij,kl->ikjl"(σx,σx)-
      Jy*ein"ij,kl->ikjl"(σy,σy)
    
    h=real(h); h=reshape(h,4,4); h=exp(τ*h)
    
    h=reshape(h,(2,2,2,2)); h=permutedims(h,(1,3,2,4)); h=reshape(h,(4,4))

    U,S,V=svd(h); S=sqrt.(S)
    U=U*diagm(S); U=reshape(U,(2,2,4))
    V=V*diagm(S); V=reshape(V,(2,2,4))

    AL,AR = permutedims(U,(3,1,2)), permutedims(V,(3,1,2))    

    A=ein"iαγ,jγβ->ijαβ"(AR,AL)   
    
    return AL,A,AR
end