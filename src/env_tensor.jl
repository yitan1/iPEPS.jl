"""
    EnvTensor

1. corner: C1, C2, C3, C4; 
    - dim(C): χ*χ
    ```
        C -- 2
        |
        1
    ```
2. edge: E1, E2, E3, E4; 
    - dim(E): χ*D²*χ
    ```
        1
        |
        E -- 3
        |
        2
    ```
"""
struct EnvTensor{T, CT<:AbstractArray{T}, ET<:AbstractArray{T}}
    corner::Vector{CT}
    edge::Vector{ET}
end

EnvTensor(corner::Vector{CT}, edge::Vector{ET}) where {T, CT<:AbstractArray{T}, ET<:AbstractArray{T}} = EnvTensor{T, CT, ET}(corner, edge)

corner(env::EnvTensor) = env.corner
edge(env::EnvTensor) = env.edge

# C1 -- E1 -- C2
# |     |     |
# E4 -- AA -- E2
# |     |     |
# C4 -- E3 -- C3
function get_envtensor(phi::PEPS; kwargs...)
    env0 = init_env(phi)
    env = update_env(env0, phi; kwargs...)

    env
end

function init_env(phi::IPEPS)
    A = phi.A
    T = transfer_matrix(A)

    C1 = sum(T, dims = (1,2)) |> x -> dropdims(x, dims = (1,2))
    C2 = sum(T, dims = (1,4)) |> x -> dropdims(x, dims = (1,4))
    C3 = sum(T, dims = (3,4)) |> x -> dropdims(x, dims = (3,4))
    C4 = sum(T, dims = (2,3)) |> x -> dropdims(x, dims = (2,3))

    E1 = sum(T, dims = 1) |> x -> dropdims(x, dims = 1)
    E2 = sum(T, dims = 4) |> x -> dropdims(x, dims = 4)
    E3 = sum(T, dims = 3) |> x -> dropdims(x, dims = 3)
    E4 = sum(T, dims = 2) |> x -> dropdims(x, dims = 2)

    Cs = [C1, C2, C3, C4] ./ norm.([C1, C2, C3, C4])
    Es = [E1, E2, E3, E4] ./ norm.([E1, E2, E3, E4])
    
    env = EnvTensor(Cs, Es)
    env
end
    
function update_env!(env::EnvTensor, phi::IPEPS)
    env = up_left!(env, phi)
    env = up_right!(env, phi)
    env = up_top!(env, phi)
    env = up_bottom!(env, phi)

    env, cutoff
end

function up_left!(env::EnvTensor, phi::IPEPS)
    Pl, Pld, cut_off = left_projector(env, phi) 
    Cs = corner(env)
    C1, C4 = Cs[1], Cs[4]
    Es = edge(env)
    E4 = Es[4]
    A = data(phi)
    T = transfer_matrix(A)

    newC1, newE4, newC4 = proj_left(Pl, Pld, C1, E1, C4, E4, T) # XXX: unnecessay computation

    Cs[1], Es[4], Cs[4] = newC1, newE4, newC4  # change input
    env, cut_off
end

function proj_left(Pl, Pld, C1, E1, C4, E4, T)
    newC1 = C1*E1*Pl
    newE4 = Pl*E4*T*Pld
    newC4 = Pld*C4*E4

    newC1, newE4, newC4
end

function left_projector(env, phi)
    Cs = corner(env)
    Es = edge(env)
    C1, C4 = Cs[1], Cs[4]
    E1, E3, E4 = Es[1], Es[3], Es[4]
    A = data(phi)
    T = transfer_matrix(A)

    # XXX: add = QR decompisiton 
    UL = contract_ul_env(C1,E1,E4,T)
    BL = contract_bl_env(C4,E3,E4,T)
    R1 = permutedims(UL, (2,1))
    R2 = BL

    get_projector(R1, R2, chi)
end

function get_projector(R1, R2, chi)
    # BUG: potentional; should be size(R2,2)
    new_chi = min(chi, size(R1,2))  
    U, S, V = svd(R1*R2)
    ####### cut off
    U1 = U[:, 1:new_chi]
    V1 = V[:, 1:new_chi]
    
    cut_off = sum(S[new_chi+1:end]) / sum(S)   # XXX: imporve cut-off

    inv_sqrt_S = sqrt.(S[1:new_chi]) |> diagm |> inv

    P1 = R2*V1*inv_sqrt_S    #(D2,D3)*(D3,chi)*(chi,chi) --> (D2, chi) 
    P2 = inv_sqrt_S*U1'*R1   #(chi,chi)*(chi,D1)*(D1,D2)  --> (chi, D2)

    P1, P2, cut_off
end


############ 

"""
    contract_ul_env(C1,E1,E4,T)

contraction order:
```
C1 --1-- E1 -- -3
|        |
2        3
|        |
E4 --4-- T --- -4 
|        |
-1       -2
```
"""
function contract_ul_env(C1,E1,E4,T)
    @tensor UL[m1,m2,m3,m4] := C1[p2,p1]*E1[p1,p3,m3]*E4[p2,m1,p4]*T[p3,p4,m2,m4]
    UL = reshape(UL, size(UL,1)*size(UL,2), :)
    UL
end

"""
    contract_ur_env(C2,E1,E2,T)

contraction order:
```
-1 -- E1 --1-- C2
      |        |
      3        2
      |        |
-2 -- T ---4-- E2 
      |        |
     -4       -3
```
"""
function contract_ur_env(C2,E1,E2,T)
    @tensor UR[m1,m2,m3,m4] := C2[p1,p2]*E1[m1,p3,p1]*E2[p2,p4,m3]*T[p3,m2,m4,p4]
    UR = reshape(UR, size(UR,1)*size(UR,2), :)
    UR
end

"""
    contract_bl_env(C4,E3,E4,T)

contraction order:
```
-3       -4
|        |
E4 --3-- T --- -2 
|        |
2        4
|        |
C4 --1-- E3 -- -1
```
"""
function contract_bl_env(C4,E3,E4,T)
    @tensor BL[m1,m2,m3,m4] = C4[p1,p2]*E3[p1,m1,p4]*E4[m3,p2,p3]*T[m4,p3,p4,m2]
    BL = reshape(BL, size(BL,1)*size(BL,2), :)
    BL
end

"""
    contract_br_env(C3,E3,E2,T)

contraction order:
```
     -2       -1
      |        |
-4 -- T ---4-- E2 
      |        |
      3        2
      |        |
-3 -- E3 --1-- C3
```
"""
function contract_br_env(C3,E3,E2,T)
    @tensor BR[m1,m2,m3,m4] := C3[p2,p1]*E3[p3,m3,p1]*E2[m1,p4,p2]*T[m2,m4,p3,p4]
    BR = reshape(BR, size(BR,1)*size(BR,2), :)
    BR
end

