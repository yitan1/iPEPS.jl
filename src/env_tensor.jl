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
struct EnvTensor{T, N, AT<:AbstractArray{T,N}, CT<:AbstractArray{T}, ET<:AbstractArray{T}}
    bulk::AT
    corner::Vector{CT}
    edge::Vector{ET}
    chi::Int
end

EnvTensor(bulk::AT, corner::Vector{CT}, edge::Vector{ET}, chi) where {T, N, AT<:AbstractArray{T,N}, CT<:AbstractArray{T}, ET<:AbstractArray{T}} = EnvTensor{T, N, AT, CT, ET}(bulk, corner, edge)

Base.eltype(::Type{<:EnvTensor{T}}) where {T} = T

bulk(env::EnvTensor) = env.bulk
corner(env::EnvTensor) = env.corner
edge(env::EnvTensor) = env.edge

get_maxchi(env::EnvTensor) = env.chi 

"""
    get_envtensor

```
C1 -- E1 -- C2
|     |     |
E4 -- T  -- E2
|     |     |
C4 -- E3 -- C3
```
"""
function get_envtensor(phi::PEPS; kwargs...)
    chi = get(kwargs, :chi, 100) # TODO: error when chi is not assigned

    env = init_env(phi, chi)

    maxitr = get(kwargs, :maxitr, 1000)
    conv_tol = get(kwargs, :conv_tol, 1e-8)
    olds = zeros(eltype(env), chi)
    diff = 1.0

    for i = 1:maxitr
        s = update_env!(env)

        if length(s) == length(olds)
            diff = norm(s - olds)
            @show i, diff
        end
        if diff < conv_tol
            break
        end
        olds = s
    end

    env
end

function init_env(phi::IPEPS, chi)
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
    
    env = EnvTensor(T, Cs, Es, chi)
    env
end
    
function update_env!(env::EnvTensor; kwargs...)
    s1 = up_left!(env)
    __ = up_right!(env)
    __ = up_top!(env)
    __ = up_bottom!(env)

    s1
end

"""
    up_left!

update (C1, E4, C4)
"""
function up_left!(env::EnvTensor)
    Pl, Pld, s = left_projector(env) 
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    newC1, newE4, newC4 = proj_left(Pl, Pld, Cs[1], Es[1], Es[4], T, Cs[4], Es[3]) # XXX: unnecessay computation

    Cs[1], Es[4], Cs[4] = newC1, newE4, newC4  # change input variable 
    s
end

"""
    up_right!

update (C2, E2, C3)
"""
function up_right!(env::EnvTensor)
    Pr, Prd, s = right_projector(env) 
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    newC2, newE2, newC3 = proj_right(Pr, Prd, Cs[2], Es[1], Es[2], T, Cs[3], Es[3]) # XXX: unnecessay computation

    Cs[2], Es[2], Cs[3] = newC2, newE2, newC3  # change input variable 
    s
end

"""
    up_top!

update (C1, E1, C2)
"""
function up_top!(env::EnvTensor) 
    Pt, Ptd, s = top_projector(env)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    newC1, newE1, newC2 = proj_top(Pt, Ptd, Cs[1], Es[4], Es[1], T, Cs[2], Es[2]) # XXX: unnecessay computation

    Cs[1], Es[1], Cs[2] = newC1, newE1, newC2  # change input variable 
    s
end

"""
    up_bottom!

update (C4, E3, C3)
"""
function up_bottom!(env::EnvTensor) 
    Pb, Pbd, s = bottom_projector(env)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    newC4, newE3, newC3 = proj_bottom(Pb, Pbd, Cs[4], Es[4], Es[3], T, Cs[3], Es[2]) # XXX: unnecessay computation

    Cs[4], Es[3], Cs[3] = newC4, newE3, newC3  # change input variable 
    s
end

function left_projector(env::EnvTensor)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    # XXX: add = QR decompisiton 
    UL = contract_ul_env(Cs[1],Es[1],Es[4],T)
    BL = contract_bl_env(Cs[4],Es[3],Es[4],T)
    R1 = permutedims(UL, (2,1))
    R2 = BL

    chi = get_maxchi(env)
    get_projector(R1, R2, chi)
end

function right_projector(env::EnvTensor) 
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    # XXX: add = QR decompisiton 
    UR = contract_ur_env(Cs[2],Es[1],Es[2],T)
    BR = contract_br_env(Cs[3],Es[3],Es[2],T)
    R1 = UR
    R2 = BR

    chi = get_maxchi(env)
    get_projector(R1, R2, chi)
end

function top_projector(env::EnvTensor)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    # XXX: add = QR decompisiton 
    UL = contract_ul_env(Cs[1],Es[1],Es[4],T)
    UR = contract_ur_env(Cs[2],Es[1],Es[2],T)
    R1 = UL
    R2 = UR

    chi = get_maxchi(env)
    get_projector(R1, R2, chi)
end

function bottom_projector(env::EnvTensor)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    # XXX: add = QR decompisiton 
    BL = contract_bl_env(Cs[4],Es[3],Es[4],T)
    BR = contract_br_env(Cs[3],Es[3],Es[2],T)
    R1 = BL
    R2 = permutedims(BR,(2,1))

    chi = get_maxchi(env)
    get_projector(R1, R2, chi)
end

function get_projector(R1, R2, chi)
    # BUG: potentional; should be size(R2,2)
    new_chi = min(chi, size(R1,2))  
    U, S, V = svd(R1*R2)
    ####### cut off
    U1 = U[:, 1:new_chi]
    V1 = V[:, 1:new_chi]
    S = S./S[1]
    S1 = S[1:new_chi]
    
    # cut_off = sum(S[new_chi+1:end]) / sum(S)   # XXX: imporve cut-off

    inv_sqrt_S = sqrt.(S1) |> diagm |> inv

    P1 = R2*V1*inv_sqrt_S    #(D2,D3)*(D3,chi)*(chi,chi) --> (D2, chi) 
    P2 = inv_sqrt_S*U1'*R1   #(chi,chi)*(chi,D1)*(D1,D2)  --> (chi, D2)

    P1, P2, S1
end

########### 

"""
    proj_left

```
C1 -- E1 --       
  \\  /
   Pl
   | 
   |
   Pld
  /  \\
E4 -- T --
  \\  /
   Pl
   |
   |
   Pld
  /  \\
C4 -- E3 --       
```
"""
function proj_left(Pl, Pld, C1, E1, E4, T, C4, E3)
    @tensor newC1[m1,m2,m3] := C1[m1, p1]*E1[p1,m2,m3]
    newC1 = reshape(newC1, :, size(newC1,3))
    @tensor newC1[m1,m2] := newC1[p1,m2]*Pl[p1,m1]

    @tensor newE4[m1,m2,m3,m4,m5] := E4[m1,m3,p1]*T[m2,p1,m4,m5]
    newE4 = reshape(newE4, size(newE4,1)*size(newE4,2), size(newE4,3)*size(newE4,4), :)
    @tensor newE4[m1,m2,m3] := Pld[m1,p1]*newE4[p1,p2,m3]*Pl[p2,m2]
    
    @tensor newC4[m1,m2,m3] := C4[m1, p1]*E3[m2,p1,m3]
    newC4 = reshape(newC4, :, size(newC4,3))
    @tensor newC4[m1,m2] := Pld[m1,p1]*newC4[p1,m2]

    newC1/norm(newC1), newE4/norm(newE4), newC4/norm(newC4)
end

"""
    proj_right

```
-- E1 -- C2       
     \\  /
      Pr
      | 
      |
      Prd
     /  \\
--- T -- E2 
     \\  /
      Pr
      |
      |
      Prd
     /  \\
-- E3 -- C3       
```
"""
function proj_right(Pr, Prd, C2, E1, E2, T, C3, E3)
    @tensor newC2[m1,m2,m3] := E1[m1,m3,p1]*C2[p1, m2] 
    newC2 = reshape(newC2, size(newC2,1), :)
    @tensor newC2[m1,m2] := newC2[m1,p1]*Pr[p1,m2]

    @tensor newE2[m1,m2,m3,m4,m5] :=T[m2,m3,m5,p1]*E2[m1,p1,m4]
    newE2 = reshape(newE2, size(newE2,1)*size(newE2,2), :, size(newE2,4)*size(newE2,5))
    @tensor newE2[m1,m2,m3] := Prd[m1,p1]*newE2[p1,m2,p2]*Pr[p2,m3]
    
    @tensor newC3[m1,m2,m3] := E3[m2,m3,p1]*C3[m1, p1]
    newC3 = reshape(newC3, :, size(newC3,3))
    @tensor newC3[m1,m2] := Prd[m1,p1]*newC3[p1,m2]

    newC2/norm(newC2), newE2/norm(newE2), newC3/norm(newC3)
end

"""
    proj_top

```
C1 \\             /  E1 \\            / C2 
|    - Pt - Ptd -   |   - Pt - Ptd -  |
E4 /             \\  T  /            \\ E2
|                   |                 |
```
"""
function proj_top(Pt, Ptd, C1, E4, E1, T, C2, E2) 
    @tensor newC1[m1,m2,m3] := C1[p1, m2]*E4[p1, m1, m3] 
    newC1 = reshape(newC1, size(newC1,1), :)
    @tensor newC1[m1,m2] := newC1[m1,p1]*Pt[p1,m2]

    @tensor newE1[m1,m2,m3,m4,m5] :=E1[m1,p1,m4]*T[p1,m2,m3,m5]
    newE1 = reshape(newE1, size(newE1,1)*size(newE1,2), :, size(newE1,4)*size(newE1,5))
    @tensor newE1[m1,m2,m3] := Ptd[m1,p1]*newE1[p1,m2,p2]*Pt[p2,m3]
    
    @tensor newC2[m1,m2,m3] := C2[m1,p1]*E2[p1,m2,m3]
    newC2 = reshape(newC2, :, size(newC2,3))
    @tensor newC2[m1,m2] := Ptd[m1,p1]*newC2[p1,m2]

    newC1/norm(newC1), newE1/norm(newE1), newC2/norm(newC2)
end

"""
    proj_bottom

```
|                   |                 |
E4 \\             /  T  \\            / E2 
|    - Pb - Pbd -   |   - Pb - Pbd -  |
C4 /             \\  E3 /            \\ C3
```
"""
function proj_bottom(Pb, Pbd, C4, E4, E3, T, C3, E2) 
    @tensor newC4[m1,m2,m3] := E4[m1,p1,m3]*C4[p1, m2] 
    newC4 = reshape(newC4, size(newC4,1), :)
    @tensor newC4[m1,m2] := newC4[m1,p1]*Pb[p1,m2]

    @tensor newE3[m1,m2,m3,m4,m5] := T[m1,m3,p1,m5]*E3[p1,m2,m4]
    newE3 = reshape(newE3, :, size(newE3,2)*size(newE3,3), size(newE3,4)*size(newE3,5))
    @tensor newE3[m1,m2,m3] := Pbd[m2,p1]*newE3[m1,p1,p2]*Pb[p2,m3]
    
    @tensor newC3[m1,m2,m3] := E2[m1,m3,p1]*C3[p1, m2]
    newC3 = reshape(newC3, size(newC3,1), :)
    @tensor newC3[m1,m2] := Pbd[m2,p1]*newC3[m1,p1]

    newC4/norm(newC4), newE3/norm(newE3), newC3/norm(newC3)
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
-1       -2
|        |
E4 --3-- T --- -4 
|        |
2        4
|        |
C4 --1-- E3 -- -3
```
"""
function contract_bl_env(C4,E3,E4,T)
    @tensor BL[m1,m2,m3,m4] := C4[p2,p1]*E3[p4,p1,m3]*E4[m1,p2,p3]*T[m2,p3,p4,m4]
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

