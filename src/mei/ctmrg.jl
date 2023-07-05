using LinearAlgebra, TensorOperations

function ctmrgstep(dt::DoubleTensors;χ=100)
    @tensor C[a,d,b,c] := ((dt.C1[e,f]*dt.T4[a,e,g])*dt.T1[f,b,h])*dt.A[h,c,d,g]
    dt.C1,U1,V1=getProjection(C,χ)

    @tensor C[a,d,b,c] := ((dt.C2[e,f]*dt.T1[a,e,g])*dt.T2[f,b,h])*dt.A[g,h,c,d]
    dt.C2,U2,V2=getProjection(C,χ)

    @tensor C[a,d,b,c] := ((dt.C3[e,f]*dt.T2[a,e,g])*dt.T3[f,b,h])*dt.A[d,g,h,c]
    dt.C3,U3,V3=getProjection(C,χ)

    @tensor C[a,d,b,c] := ((dt.C4[e,f]*dt.T3[a,e,g])*dt.T4[f,b,h])*dt.A[c,d,g,h]
    dt.C4,U4,V4=getProjection(C,χ)

    @tensor T[v,u,f] := (V1[b,c,v]*dt.T1[b,a,e])*dt.A[e,d,f,c]*U2[a,d,u]
    dt.T1=T

    @tensor T[v,u,f] := (V2[b,c,v]*dt.T2[b,a,e])*dt.A[c,e,d,f]*U3[a,d,u]
    dt.T2=T

    @tensor T[v,u,f] := (V3[b,c,v]*dt.T3[b,a,e])*dt.A[f,c,e,d]*U4[a,d,u]
    dt.T3=T

    @tensor T[v,u,f] := (V4[b,c,v]*dt.T4[b,a,e])*dt.A[d,f,c,e]*U1[a,d,u]
    dt.T4=T
    return dt,diag(dt.C1)/dt.C1[1]
end

function getProjection(C,χ)
    dim=size(C); C=reshape(C,dim[1]*dim[2],dim[3]*dim[4])
    U,S,V=svd(C)
    χ=min(χ,dim[1]*dim[2],dim[3]*dim[4])
    C=diagm(S[1:χ])
    U=reshape(U[:,1:χ],(dim[1],dim[2],χ))
    V=reshape(V[:,1:χ],(dim[3],dim[4],χ))
    return C,U,V
end