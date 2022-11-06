"""
    EnvTensor

corner: C1, C2, C3, C4
edge: E1, E2, E3, E4

dim(C): χ*χ
dim(E): χ*D²*χ
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
    
function update_env(env0::EnvTensor, phi::IPEPS)
    env = up_left(env0, phi)
    env = up_right(env, phi)
    env = up_top(env, phi)
    env = up_bottom(env, phi)

    env, cutoff
end

function up_left(env::EnvTensor, phi::IPEPS)
    Pl, Pld = left_projector(env, phi) 
    Cs = corner(env)
    Es = edge(env)
    A = data(phi)
    T = transfer_matrix(A)

    proj_left(Pl, Pld, C1, E4, C4, T)
end

function left_projector(env, phi)
    Cs = corner(env)
    Es = edge(env)
    C1, C4 = Cs[1], Cs[4]
    E1, E3, E4 = Es[1], Es[3], Es[4]
    A = data(phi)
    T = transfer_matrix(A)

    ## !!! add = QR decompisiton 
    UL = contract_ul_env(C1,E1,E4,T)
    BL = contract_bl_env(C4,E3,E4,T)
    R1 = permutedims(UL, (2,1))
    R2 = BL

    get_projector(R1, R2, chi)
end

function get_projector(R1, R2, chi)
    # !!Warn: should be size(R2,2)
    new_chi = min(chi, size(R1,2))  
    U, S, V = svd(R1*R2)
    ####### cut off
    U1 = U[:, 1:new_chi]
    V1 = V[:, 1:new_chi]
    
    cut_off = sum(S[new_chi+1:end]) / sum(S)

    inv_sqrt_S = sqrt.(S[1:new_chi]) |> diagm |> inv

    P1 = R2*V1*inv_sqrt_S    #(D2,D3)*(D3,chi)*(chi,chi) --> (D2, chi) 
    P2 = inv_sqrt_S*U1'*R1   #(chi,chi)*(chi,D1)*(D1,D2)  --> (chi, D2)

    P1, P2
end


function contract_ul_env(C1,E1,E4,T)
    @tensor UL[:] = C1*E1*E4*T
    UL
end

function contract_bl_env(C4,E3,E4,T)
    @tensor UL[:] = C4*E3*E4*T
    UL
end

