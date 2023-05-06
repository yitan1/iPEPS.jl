using LinearAlgebra
using BenchmarkTools
using MKL
using TSVD
using TensorOperations
# using OMEinsum
export test_rg
function test_rg1()
    
end
function init_CTA(A_s)
    #A_s[1,2,3,4,5] has the index assignment. 5 is phy;

    dsize=size(A_s);
    @time @tensor A[a,e,b,f,c,g,d,h] := A_s[a,b,c,d,i]*A_s[e,f,g,h,i]
    @time @tensor C1[a,e,d,h]        := A_s[a,b,c,d,i]*A_s[e,b,c,h,i]
    @time @tensor C2[b,f,a,e]        := A_s[a,b,c,d,i]*A_s[e,f,c,d,i]
    @time @tensor C3[c,g,b,f]        := A_s[a,b,c,d,i]*A_s[a,f,g,d,i]
    @time @tensor C4[d,h,c,g]        := A_s[a,b,c,d,i]*A_s[a,b,g,h,i]
    @time @tensor T1[b,f,d,h,a,e]    := A_s[a,b,c,d,i]*A_s[e,f,c,h,i]
    @time @tensor T2[c,g,a,e,b,f]    := A_s[a,b,c,d,i]*A_s[e,f,g,d,i]
    @time @tensor T3[d,h,b,f,c,g]    := A_s[a,b,c,d,i]*A_s[a,f,g,h,i]
    @time @tensor T4[a,e,c,g,d,h]    := A_s[a,b,c,d,i]*A_s[e,b,g,h,i]
    A = reshape(A,(dsize[1]*dsize[1],dsize[2]*dsize[2],dsize[3]*dsize[3],dsize[4]*dsize[4]))
    C1 = reshape(C1,(dsize[1]*dsize[1],dsize[4]*dsize[4]))
    C2 = reshape(C2,(dsize[1]*dsize[1],dsize[2]*dsize[2]))
    C3 = reshape(C3,(dsize[3]*dsize[3],dsize[2]*dsize[2]))
    C4 = reshape(C4,(dsize[4]*dsize[4],dsize[3]*dsize[3]))
    T1 = reshape(T1,(dsize[2]*dsize[2],dsize[4]*dsize[4],dsize[1]*dsize[1]))
    T2 = reshape(T2,(dsize[3]*dsize[3],dsize[1]*dsize[1],dsize[2]*dsize[2]))
    T3 = reshape(T3,(dsize[4]*dsize[4],dsize[2]*dsize[2],dsize[3]*dsize[3]))
    T4 = reshape(T4,(dsize[1]*dsize[1],dsize[3]*dsize[3],dsize[4]*dsize[4]))

    return C1,C2,C3,C4,T1,T2,T3,T4,A

    
end

function test_rg()
    X = 100
    D2 = 16
    D = 4
    d = 2
    # C1, C2, C3, C4 = rand(X,X), rand(X,X), rand(X,X), rand(X,X)
    # E1, E2, E3, E4 = rand(X, X, D2), rand(X, X, D2), rand(X, X, D2), rand(X, X, D2)
    # T = rand(D2, D2, D2, D2) |> normalize
    # C1, C2, C3, C4 = normalize(C1), normalize(C2), normalize(C3), normalize(C4)
    # E1, E2, E3, E4 = normalize(E1), normalize(E2), normalize(E3), normalize(E4)
    A = rand(D, D, D, D, d)
    C1,C2,C3,C4,E1,E2,E3,E4, T  = init_CTA(A)
    @show X, D2
    get_envtensor(T, chi = X)
    # maxitr = 100
    # diff = 1
    
    # @time C1,C2,C3,C4,E1,E2,E3,E4 = getCTA(C1,C2,C3,C4,E1,E2,E3,E4,T,X)
    # for i = 1:maxitr
    #     oldC1 = C1
    #     @time C1,C2,C3,C4,E1,E2,E3,E4 = getCTA(C1,C2,C3,C4,E1,E2,E3,E4,T,X)
    #     diff = norm(C1-oldC1)
    #     @show diff
    #     if diff < 1e-7
    #         @show diff
    #         println("conv")
    #         for i = 1:4
    #             oldC1 = C1
    #             @time C1,C2,C3,C4,E1,E2,E3,E4 = getCTA(C1,C2,C3,C4,E1,E2,E3,E4,T,X)
    #             diff = maximum(C1-oldC1)
    #             @show diff
    #         end
    #         break
    #     end
    # end

    # C1,C2,C3,C4,E1,E2,E3,E4
end
function getCTA(C1,C2,C3,C4,T1,T2,T3,T4,A,chi)
    # C1, C2, C3, C4, T1, T2, T3, T4 and A have index assignments as in tensor_index.jpg
    # chi: trucation dimension

    # get C1,C2,C3,C4 and projections
    # tensor indices [a,b,c,d,e,f,g,h] are assigned as in svd_C1.jpg
    # println("TensorOperations")
    @tensor CTTA1[a,d,b,c] := ((C1[e,f]*T4[a,e,g])*T1[f,b,h])*A[d,g,h,c]
    @tensor CTTA2[a,d,b,c] := ((C2[e,f]*T1[a,e,g])*T2[f,b,h])*A[c,d,g,h]
    @tensor CTTA3[a,d,b,c] := ((C3[e,f]*T2[a,e,g])*T3[f,b,h])*A[h,c,d,g]
    @tensor CTTA4[a,d,b,c] := ((C4[e,f]*T3[a,e,g])*T4[f,b,h])*A[g,h,c,d]
    # @time @tensor CTTA1[a,d,b,c] := ((C1[e,f]*T4[a,e,g])*T1[f,b,h])*A[d,g,h,c]
    # @time @tensor CTTA2[a,d,b,c] := ((C2[e,f]*T1[a,e,g])*T2[f,b,h])*A[c,d,g,h]
    # @time @tensor CTTA3[a,d,b,c] := ((C3[e,f]*T2[a,e,g])*T3[f,b,h])*A[h,c,d,g]
    # @time @tensor CTTA4[a,d,b,c] := ((C4[e,f]*T3[a,e,g])*T4[f,b,h])*A[g,h,c,d]
    # println("OMEinsum")
    # @time @ein CTTA1[a,d,b,c] := ((C1[e,f]*T4[a,e,g])*T1[f,b,h])*A[d,g,h,c]
    # @time @ein CTTA2[a,d,b,c] := ((C2[e,f]*T1[a,e,g])*T2[f,b,h])*A[c,d,g,h]
    # @time @ein CTTA3[a,d,b,c] := ((C3[e,f]*T2[a,e,g])*T3[f,b,h])*A[h,c,d,g]
    # @time @ein CTTA4[a,d,b,c] := ((C4[e,f]*T3[a,e,g])*T4[f,b,h])*A[g,h,c,d]

    # after contraction, we perform the svd. [svd_C1.jpg]
    # println("projections")
    newC1,adU1,bcV1=getProjection(CTTA1,chi)
    newC2,adU2,bcV2=getProjection(CTTA2,chi)
    newC3,adU3,bcV3=getProjection(CTTA3,chi)
    newC4,adU4,bcV4=getProjection(CTTA4,chi)
    # println("TensorOperations")
    # get T1 after projections    
    @tensor newT1[v,u,f] := ((bcV1[b,c,v]*T1[b,a,e])*A[f,c,e,d])*adU2[a,d,u]
    @tensor newT2[v,u,f] := ((bcV2[b,c,v]*T2[b,a,e])*A[d,f,c,e])*adU3[a,d,u]
    @tensor newT3[v,u,f] := ((bcV3[b,c,v]*T3[b,a,e])*A[e,d,f,c])*adU4[a,d,u]
    @tensor newT4[v,u,f] := ((bcV4[b,c,v]*T4[b,a,e])*A[c,e,d,f])*adU1[a,d,u]
    # @time @tensor newT1[v,u,f] := ((bcV1[b,c,v]*T1[b,a,e])*A[f,c,e,d])*adU2[a,d,u]
    # @time @tensor newT2[v,u,f] := ((bcV2[b,c,v]*T2[b,a,e])*A[d,f,c,e])*adU3[a,d,u]
    # @time @tensor newT3[v,u,f] := ((bcV3[b,c,v]*T3[b,a,e])*A[e,d,f,c])*adU4[a,d,u]
    # @time @tensor newT4[v,u,f] := ((bcV4[b,c,v]*T4[b,a,e])*A[c,e,d,f])*adU1[a,d,u]
    # println("OMEinsum")
    # @time @ein newT1[v,u,f] := ((bcV1[b,c,v]*T1[b,a,e])*A[f,c,e,d])*adU2[a,d,u]
    # @time @ein newT2[v,u,f] := ((bcV2[b,c,v]*T2[b,a,e])*A[d,f,c,e])*adU3[a,d,u]
    # @time @ein newT3[v,u,f] := ((bcV3[b,c,v]*T3[b,a,e])*A[e,d,f,c])*adU4[a,d,u]
    # @time @ein newT4[v,u,f] := ((bcV4[b,c,v]*T4[b,a,e])*A[c,e,d,f])*adU1[a,d,u]
    return normalize(newC1), normalize(newC2), normalize(newC3), normalize(newC4), normalize(newT1), normalize(newT2), normalize(newT3), normalize(newT4)
end

function getProjection(CTTA,dchi)
    dsize = size(CTTA)
    dchi=min(dchi,dsize[1]*dsize[2],dsize[3]*dsize[4])

    # if min(dsize[1]*dsize[2],dsize[3]*dsize[4])>dchi+30
        # adU,S,bcV=tsvd(reshape(CTTA,(dsize[1]*dsize[2],dsize[3]*dsize[4])),dchi+20)    
    # else
        adU,S,bcV= svd(reshape(CTTA,(dsize[1]*dsize[2],dsize[3]*dsize[4])))    
    # end
    dchi=count(>=(S[dchi]-1.0E-12),S)    
    C  =diagm(S[1:dchi])
    adU=reshape(adU[:,1:dchi],(dsize[1],dsize[2],dchi))
    bcV=reshape(bcV[:,1:dchi],(dsize[3],dsize[4],dchi))
    return C,adU,bcV
end

function normalize(A)
    max = maximum(abs.(A))
    A = A./max
    A
end