mutable struct DoubleTensors{T,CT<:AbstractArray{T,2},ET<:AbstractArray{T,3},BT<:AbstractArray{T,4}} 
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

function init_DoubleTensor(As::AbstractArray{T,5}) where T
    @tensor C1[c,k,b,j] := As[a,b,c,d,α]*As[a,j,k,d,α]    
    dim=size(C1); C1=reshape(C1,dim[1]*dim[2],dim[3]*dim[4])

    @tensor C2[d,l,c,k] := As[a,b,c,d,α]*As[a,b,k,l,α]    
    dim=size(C2); C2=reshape(C2,dim[1]*dim[2],dim[3]*dim[4])

    @tensor C3[a,i,d,l] := As[a,b,c,d,α]*As[i,b,c,l,α]
    dim=size(C3); C3=reshape(C3,dim[1]*dim[2],dim[3]*dim[4])

    @tensor C4[b,j,a,i] := As[a,b,c,d,α]*As[i,j,c,d,α]
    dim=size(C4); C4=reshape(C4,dim[1]*dim[2],dim[3]*dim[4])

    @tensor T1[d,l,b,j,c,k] := As[a,b,c,d,α]*As[a,j,k,l,α]
    dim=size(T1); T1=reshape(T1,dim[1]*dim[2],dim[3]*dim[4],dim[5]*dim[6])

    
    @tensor T2[a,i,c,k,d,l] := As[a,b,c,d,α]*As[i,b,k,l,α]    
    dim=size(T2); T2=reshape(T2,dim[1]*dim[2],dim[3]*dim[4],dim[5]*dim[6])

    @tensor T3[b,j,d,l,a,i] := As[a,b,c,d,α]*As[i,j,c,l,α]    
    dim=size(T3); T3=reshape(T3,dim[1]*dim[2],dim[3]*dim[4],dim[5]*dim[6])

    @tensor T4[c,k,a,i,b,j] := As[a,b,c,d,α]*As[i,j,k,d,α]
    dim=size(T4); T4=reshape(T4,dim[1]*dim[2],dim[3]*dim[4],dim[5]*dim[6])

    @tensor A[a,i,b,j,c,k,d,l] := As[a,b,c,d,α]*As[i,j,k,l,α]
    dim=size(A); A=reshape(A,dim[1]*dim[2],dim[3]*dim[4],dim[5]*dim[6],dim[7]*dim[8])

    return C1,C2,C3,C4,T1,T2,T3,T4,A
end

function init_DoubleTensor(As::AbstractArray{T,5},pepo::PEPOTensors) where T
    @tensor C1[c,α,k,b,β,j] := As[a,b,c,d,γ]*As[a,j,k,d,δ]*pepo.C1[α,β,γ,δ]    
    dim=size(C1); C1=reshape(C1,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6])

    @tensor C2[d,α,l,c,β,k] := As[a,b,c,d,γ]*As[a,b,k,l,δ]*pepo.C2[α,β,γ,δ]    
    dim=size(C2); C2=reshape(C2,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6])

    @tensor C3[a,α,i,d,β,l] := As[a,b,c,d,γ]*As[i,b,c,l,δ]*pepo.C3[α,β,γ,δ]    
    dim=size(C3); C3=reshape(C3,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6])

    @tensor C4[b,α,j,a,β,i] := As[a,b,c,d,γ]*As[i,j,c,d,δ]*pepo.C4[α,β,γ,δ]
    dim=size(C4); C4=reshape(C4,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6])

    @tensor T1[d,α,l,b,β,j,c,γ,k] := As[a,b,c,d,δ]*As[a,j,k,l,ϵ]*pepo.T1[α,β,γ,δ,ϵ]    
    dim=size(T1); T1=reshape(T1,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6],dim[7]*dim[8]*dim[9])

    
    @tensor T2[a,α,i,c,β,k,d,γ,l] := As[a,b,c,d,δ]*As[i,b,k,l,ϵ]*pepo.T2[α,β,γ,δ,ϵ]    
    dim=size(T2); T2=reshape(T2,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6],dim[7]*dim[8]*dim[9])

    @tensor T3[b,α,j,d,β,l,a,γ,i] := As[a,b,c,d,δ]*As[i,j,c,l,ϵ]*pepo.T3[α,β,γ,δ,ϵ]    
    dim=size(T3); T3=reshape(T3,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6],dim[7]*dim[8]*dim[9])

    @tensor T4[c,α,k,a,β,i,b,γ,j] := As[a,b,c,d,δ]*As[i,j,k,d,ϵ]*pepo.T4[α,β,γ,δ,ϵ]
    dim=size(T4); T4=reshape(T4,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6],dim[7]*dim[8]*dim[9])

    @tensor A[a,α,i,b,β,j,c,γ,k,d,δ,l] := As[a,b,c,d,ϵ]*As[i,j,k,l,ζ]*pepo.A[α,β,γ,δ,ϵ,ζ]    
    dim=size(A); A=reshape(A,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6],dim[7]*dim[8]*dim[9],dim[10]*dim[11]*dim[12])

    return C1,C2,C3,C4,T1,T2,T3,T4,A
end

function getAh(As::AbstractArray{T,5},pepo::PEPOTensors) where T
    @tensor Ah[a,α,b,β,c,γ,d,δ,ζ] := As[a,b,c,d,ϵ]*pepo.A[α,β,γ,δ,ϵ,ζ]    
    dim=size(Ah); Ah=reshape(Ah,dim[1]*dim[2],dim[3]*dim[4],dim[5]*dim[6],dim[7]*dim[8],dim[9])
    return Ah
end

function getdt(As::AbstractArray{T,5},pepo::PEPOTensors) where T
    @tensor A[a,α,i,b,β,j,c,γ,k,d,δ,l] := As[a,b,c,d,ϵ]*As[i,j,k,l,ζ]*pepo.A[α,β,γ,δ,ϵ,ζ]    
    dim=size(A); A=reshape(A,dim[1]*dim[2]*dim[3],dim[4]*dim[5]*dim[6],dim[7]*dim[8]*dim[9],dim[10]*dim[11]*dim[12])
    return A
end

function getdt(As::AbstractArray{T,5}) where T
    @tensor A[a,i,b,j,c,k,d,l] := As[a,b,c,d,α]*As[i,j,k,l,α]
    dim=size(A); A=reshape(A,dim[1]*dim[2],dim[3]*dim[4],dim[5]*dim[6],dim[7]*dim[8])
    return A
end

function envNormalize(dth::DoubleTensors,dtn::DoubleTensors)
    dth.C1/=norm(dtn.C1)
    dtn.C1/=norm(dtn.C1)

    dth.C2/=norm(dtn.C2)
    dtn.C2/=norm(dtn.C2)

    dth.C3/=norm(dtn.C3)
    dtn.C3/=norm(dtn.C3)

    dth.C4/=norm(dtn.C4)
    dtn.C4/=norm(dtn.C4)

    dth.T1/=norm(dtn.T1)
    dtn.T1/=norm(dtn.T1)

    dth.T2/=norm(dtn.T2)
    dtn.T2/=norm(dtn.T2)

    dth.T3/=norm(dtn.T3)
    dtn.T3/=norm(dtn.T3)

    dth.T4/=norm(dtn.T4)
    dtn.T4/=norm(dtn.T4)
    return dth, dtn
end