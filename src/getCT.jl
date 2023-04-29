using LinearAlgebra
using BenchmarkTools
using MKL
using TSVD
using TensorOperations
# using OMEinsum
export test_rg
function test_rg()
    X = 100
    D2 = 16
    C1, C2, C3, C4 = rand(X,X), rand(X,X), rand(X,X), rand(X,X)
    E1, E2, E3, E4 = rand(X, X, D2), randn(X, X, D2), rand(X, X, D2), rand(X, X, D2)
    T = rand(D2, D2, D2, D2) |> normalize
    C1, C2, C3, C4 = normalize(C1), normalize(C2), normalize(C3), normalize(C4)
    E1, E2, E3, E4 = normalize(E1), normalize(E2), normalize(E3), normalize(E4)
    @show X, D2

    maxitr = 100
    diff = 1
    for i = 1:maxitr
        oldC1 = C1
        @time C1,C2,C3,C4,E1,E2,E3,E4 = getCTA(C1,C2,C3,C4,E1,E2,E3,E4,T,X)
        diff = maximum(C1-oldC1)
        @show diff
        if diff < 1e-10
            @show diff
            break
        end
    end

    C1,C2,C3,C4,E1,E2,E3,E4
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

    if min(dsize[1]*dsize[2],dsize[3]*dsize[4])>dchi+30
        adU,S,bcV=tsvd(reshape(CTTA,(dsize[1]*dsize[2],dsize[3]*dsize[4])),dchi+20)    
    else
        adU,S,bcV= svd(reshape(CTTA,(dsize[1]*dsize[2],dsize[3]*dsize[4])))    
    end
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