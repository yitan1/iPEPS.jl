"""
    EnvTensor

1. bulk: AAd
2. corner: C1, C2, C3, C4; 
    - dim(C): χ*χ
    ```
        C -- 2
        |
        1
    ```
3. edge: E1, E2, E3, E4; 
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

# T1 > T 
function EnvTensor(bulk::AT, corner::Vector{CT}, edge::Vector{ET}, chi) where {T, T1, N, 
                                AT<:AbstractArray{T,N}, CT<:AbstractArray{T1}, ET<:AbstractArray{T1}} 
    newT = promote_type(T, T1)
    newbulk = convert.(newT, bulk)
    EnvTensor(newbulk, corner, edge, chi)
end

EnvTensor(bulk::AT, corner::Vector{CT}, edge::Vector{ET}, chi) where {T, N, AT<:AbstractArray{T,N}, CT<:AbstractArray{T}, ET<:AbstractArray{T}} = EnvTensor{T, N, AT, CT, ET}(bulk, corner, edge, chi)

Base.eltype(::Type{<:EnvTensor{T}}) where {T} = T

bulk(env::EnvTensor) = env.bulk
corner(env::EnvTensor) = env.corner
edge(env::EnvTensor) = env.edge

get_maxchi(env::EnvTensor) = env.chi 

"""
    get_envtensor

kwargs: chi = 100, maxitr = 1000, conv_tol = 1e-8, output = true, inplace = false
```
C1 -- E1 -- C2
|     |     |
E4 -- T  -- E2
|     |     |
C4 -- E3 -- C3
```
"""
function get_envtensor(phi::IPEPS; kwargs...) 
    T = transfer_matrix(get_A(phi), get_Ad(phi))
    get_envtensor(T; kwargs)
end

function get_envtensor(phi::IPEPS, phid::IPEPS; kwargs...) 
    T = transfer_matrix(get_A(phi), get_Ad(phid))
    get_envtensor(T; kwargs)
end

function get_envtensor(T::AbstractArray; kwargs...) 
    chi = get(kwargs, :chi, 10)
    env = init_env(T, chi)

    maxitr = get(kwargs, :maxitr, 1000)
    conv_tol = get(kwargs, :conv_tol, 1e-8)
    olds = zeros(eltype(env), chi)
    diff = 1.0

    output = get(kwargs, :output, false)
    inplace = get(kwargs, :inplace, false)
    for i = 1:maxitr
        if inplace == true
            s = update_env!(env)  
        else 
            env, s = update_env(env)
        end

        if length(s) == length(olds)
            diff = norm(s - olds) # XXX: imporve cut-off
            if output == true
                @show i, diff
            end
        end
        if diff < conv_tol
            break
        end
        olds = s
    end

    env
end

function init_env(phi::IPEPS, chi)
    T = transfer_matrix(get_A(phi), get_Ad(phi))
    init_env(T, chi)
end

function init_env(phi::IPEPS, phid::IPEPS, chi)
    T = transfer_matrix(get_A(phi), get_Ad(phid))
    init_env(T, chi)
end

function init_env(T::AbstractArray, chi)
    Cs, Es = init_CE(size(T,1))
    env = EnvTensor(T, Cs, Es, chi)
    env
end

function init_CE(D::Int)
    Cs = [rand(D, D) for i = 1:4]
    Es = [rand(D, D, D) for i = 1:4]

    Cs = Cs ./ norm.(Cs)
    Es = Es ./ norm.(Es)

    Cs, Es
end

function init_CE(T::AbstractArray)
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

    Cs,Es
end

function update_env(env::EnvTensor; kwargs...) 
    env, s1 = up_left(env)
    env, __ = up_right(env)
    env, __ = up_top(env)
    env, __ = up_bottom(env)

    env, s1
end

function update_env!(env::EnvTensor; kwargs...) #XXX inplace update
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

function up_left(env::EnvTensor)
    Pl, Pld, s = left_projector(env) 
    up_left(env, Pl, Pld), s
end
function up_left(env::EnvTensor, Pl, Pld)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    newC1, newE4, newC4 = proj_left(Pl, Pld, Cs[1], Es[1], Es[4], T, Cs[4], Es[3]) # XXX: unnecessay computation

    env1 = EnvTensor(T, [newC1, Cs[2], Cs[3], newC4], [Es[1], Es[2], Es[3], newE4], get_maxchi(env))
    env1
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

function up_right(env::EnvTensor)
    Pr, Prd, s = right_projector(env) 
    up_right(env, Pr, Prd), s
end
function up_right(env::EnvTensor, Pr, Prd)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    newC2, newE2, newC3 = proj_right(Pr, Prd, Cs[2], Es[1], Es[2], T, Cs[3], Es[3]) # XXX: unnecessay computation
    #TODO
    env1 = EnvTensor(T, [Cs[1], newC2, newC3, Cs[4]], [Es[1], newE2, Es[3], Es[4]], get_maxchi(env))
    env1
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

function up_top(env::EnvTensor) 
    Pt, Ptd, s = top_projector(env)
    up_top(env, Pt, Ptd), s
end
function up_top(env::EnvTensor, Pt, Ptd)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    newC1, newE1, newC2 = proj_top(Pt, Ptd, Cs[1], Es[4], Es[1], T, Cs[2], Es[2]) # XXX: unnecessay computation

    env1 = EnvTensor(T, [newC1, newC2, Cs[3], Cs[4]], [newE1, Es[2], Es[3], Es[4]], get_maxchi(env))
    env1
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

function up_bottom(env::EnvTensor) 
    Pb, Pbd, s = bottom_projector(env)
    up_bottom(env, Pb, Pbd), s
end
function up_bottom(env::EnvTensor, Pb, Pbd)
    Cs = corner(env)
    Es = edge(env)
    T = bulk(env)

    newC4, newE3, newC3 = proj_bottom(Pb, Pbd, Cs[4], Es[4], Es[3], T, Cs[3], Es[2]) # XXX: unnecessay computation

    env1 = EnvTensor(T, [Cs[1], Cs[2], newC3, newC4], [Es[1], Es[2], newE3, Es[4]], get_maxchi(env))
    env1
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
    
    # cut_off = sum(S[new_chi+1:end]) / sum(S)   

    inv_sqrt_S = sqrt.(S1) |> diagm |> inv

    P1 = R2*V1*inv_sqrt_S    #(D2,D3)*(D3,chi)*(chi,chi) --> (D2, chi) 
    P2 = inv_sqrt_S*U1'*R1   #(chi,chi)*(chi,D1)*(D1,D2)  --> (chi, D2)

    P1, P2, S1
end


