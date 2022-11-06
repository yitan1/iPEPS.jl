using LinearAlgebra
# using Tullio
using TensorOperations
using TensorRules

# A[phy, up, left, down, right]
# C[down, right]
# E[up,right, down]
# T[up, left, down, right]
@∇ function CTM(A; chi = 30)
    D = size(A,2)
    @tensor T[a,e,b,f,c,g,d,h] := A[i, a,b,c,d]*conj(A)[i, e,f,g,h]
    T = reshape(T, D*D, D*D, D*D, D*D)/norm(T)
    
    C = sum(T, dims = (1,2))
    C = dropdims(C, dims = (1,2))
    E = sum(T, dims = 2)
    E = dropdims(E, dims = 2)
    E = permutedims(E, (1,3,2))
    
    delta = 1e-10 #TODO
    chi = chi     #TODO
    maxitr = 500  #TODO
    err_sum = 0.0
    sold = zeros(Float64,chi)
    diff = 1.0
    for n = 1:maxitr
        C, E, s, err = updateCTM(C, E, T, chi)
        
        err_sum += err
        if length(s) == length(sold)
            diff = norm(s-sold)
            # @show diff, err_sum
        end
        if diff < delta
            break
        end
        sold = s
    end
    C, E
end

@∇ function updateCTM(C,E,T,chi)
    dimE = size(E,3)
    dimT = size(T,3)
    newD = min(dimE*dimT, chi)
    
    # compute CEET
    @tensor rho[a,b,c] := C[i,a]*E[i,b,c] 
    @tensor rho[a,b,c,d] := rho[i,c,d]*E[i,a,b]
    @tensor rho[a,b,c,d] := rho[i,a, j,c]*T[i,j, d,b]
    rho = reshape(rho, dimE*dimT, dimE*dimT)
    rho = rho' + rho
    rho = rho/ norm(rho)
    
    # construct projector operator P
    U, S, V = svd(rho)
    error = sum(S[newD+1:end]) / sum(S)
    P = U[:, 1:newD]
    
    # compute new C,E
    C = P' * rho * P
    
    PE = reshape(P, dimE, dimT, newD)
    @tensor E[a,b,c,d] := E[i,d,b]*PE[i,c,a]
    @tensor E[a,b,c,d] := E[a,c,i,j]*T[i,j,d,b]
    @tensor E[a,b,c] := E[a,b,i,j]*PE[i,j,c]
    
    # symmetrize C and E
    C = C + C'
    E = E + permutedims(E,(3,2,1))
    
    S = abs.(S)
    S = S/maximum(S)
    
    C/norm(C), E/norm(E), S, error
end