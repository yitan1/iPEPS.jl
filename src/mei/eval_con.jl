using TensorOperations

function envCT(dt::DoubleTensors)
    @tensor CT[i,j,k,l] := (dt.C1[a,b]*dt.T1[b,c,i])*(dt.C2[c,d]*dt.T2[d,e,j])*
                           (dt.C3[e,f]*dt.T3[f,g,k])*(dt.C4[g,h]*dt.T4[h,a,l])
    return CT                           
end


function dt_con(dt::DoubleTensors)
    @tensor CT[i,j,k,l] := (dt.C1[a,b]*dt.T1[b,c,i])*(dt.C2[c,d]*dt.T2[d,e,j])*
                           (dt.C3[e,f]*dt.T3[f,g,k])*(dt.C4[g,h]*dt.T4[h,a,l])
    @tensor eh = CT[i,j,k,l]*dt.A[i,j,k,l]
    return eh
end

function ∂dt(dt::DoubleTensors,As::AbstractArray{T,5}) where T
    @tensor CT[i,j,k,l] := (dt.C1[a,b]*dt.T1[b,c,i])*(dt.C2[c,d]*dt.T2[d,e,j])*
                           (dt.C3[e,f]*dt.T3[f,g,k])*(dt.C4[g,h]*dt.T4[h,a,l])
    dimCT=size(CT); dimA=size(As); dim= Int.(dimCT ./ dimA[1:4])    
    CT=reshape(CT,(dimA[1],dim[1],dimA[2],dim[2],dimA[3],dim[3],dimA[4],dim[4]))
    @tensor CTA[i,j,k,l,α] := CT[i1,i,j1,j,k1,k,l1,l]*As[i1,j1,k1,l1,α]
    return CTA
end

function ∂dt(CT::AbstractArray{T,4},As::AbstractArray{T,5}) where T
    dimCT=size(CT); dimA=size(As); dim= Int.(dimCT ./ dimA[1:4])    
    CT=reshape(CT,(dimA[1],dim[1],dimA[2],dim[2],dimA[3],dim[3],dimA[4],dim[4]))
    @tensor CTA[i,j,k,l,α] := CT[i1,i,j1,j,k1,k,l1,l]*As[i1,j1,k1,l1,α]
    return CTA
end

# function ∂E(∂dth,∂dtn,eh,nn)
#     return ∂dth/nn-eh/nn*∂dtn/nn
# end

function eval_en(CTh,CTn,pepo,As)
    dthA, dtnA=getdt(As,pepo),getdt(As)

    @tensor eh=CTh[i,j,k,l]*dthA[i,j,k,l]
    @tensor nn=CTn[i,j,k,l]*dtnA[i,j,k,l]

    return eh[1]/nn[1]
end

function eval_∂E(CTh,CTn,pepo,As)
    dthA, dtnA=getdt(As,pepo),getdt(As)
    @tensor eh=CTh[i,j,k,l]*dthA[i,j,k,l]
    @tensor nn=CTn[i,j,k,l]*dtnA[i,j,k,l]

    en=eh[1]/nn[1]

    Ah=getAh(As,pepo)

    ∂eh = ∂dt(CTh,Ah)
    ∂nn = ∂dt(CTn,As)
    return ∂eh/nn[1]-en*∂nn/nn[1]
end

function eval_en(pepo,As)
    dtn=DoubleTensors(init_DoubleTensor(As)...)        

    dth=DoubleTensors(init_DoubleTensor(As,pepo)...)

    dth,dtn=envNormalize(dth,dtn)    
    CTh,CTn=envCT(dth),envCT(dtn)

    return eval_en(CTh,CTn,pepo,As)
end