
function optimize_ES(kx, ky, phi0::IPEPS, h; kwargs...) #TODO
    Bn = get_tangent_basis(phi0; kwargs)
    H, N = eff_hamitonian_norm(h, kx, ky, phi0, Bn)
    energy, _ = eigsolve(H,N)
    
    energy
end

# function get_tangent_basis(phi::ExcIPEPS)
#     get_tangent_basis(phi::IPEPS)
# end

"""
    get_tangent_basis   

Return (dD⁴-1) tensor B(d, D, D, D, D) satisfying ⟨ϕ(A)|ϕ(B)⟩ = 0
"""
function get_tangent_basis(phi::IPEPS; kwargs...)
    env = get_envtensor(phi; kwargs)
    dA = get_norm_dA(env, phi)
    dA = reshape(dA, (1,:) )
    basis = nullspace(dA)   #(dD^4, dD^4-1)  

    basis
end

"""
Return two matrix (H, N)
"""
function eff_hamitonian_norm(h, kx, ky, phi, Bn)
    M = size(Bn,2)
    H = zeros(eltype(Bn), M, M) 
    N = zeros(eltype(Bn), M, M) 
    for j in axes(H,2)
        Bj = Bn[:,j]
        Bdj = conj(Bj)
        
        hBj = gradient(_x -> effH_ij(h, kx, ky, phi, _x, Bj), Bdj)[1]
        Bj1 = gradient(_x -> effN_ij(kx, ky, phi, _x, Bj), Bdj)[1]
        for i in axes(H,1)
            Bi = Bn[:,i]
            H[i,j] = Bi'*hBj
            N[i,j] = Bi'*Bj1
        end
    end
    H, N
end

function effH_ij(h, kx, ky, phi0, Bdi, Bj)
    phi_i = ExcIPEPS(kx, ky, phi0, conj(Bdi))
    phi_j = ExcIPEPS(kx, ky, phi0, Bj)

    env = get_envtensor(phi_i, phi_j; chi = 10) # BUG
    hij = get_hor_energy(h, env, phi_i, phi_j) + get_ver_energy(h, env, phi_i, phi_j)
    hij
end

"""
    get_norm_dA

```
C1 -- E1 -- C2
|     |     |         |
E4 -- A  -- E2 =  -- dA -- 
|     |     |         | \\
C4 -- E3 -- C3
```
"""
function get_norm_dA(env::EnvTensor, phi)
    A = get_A(phi)
    Cs = corner(env)
    Es = edge(env)
    contract_env_dA(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], Es[2], Es[3], Es[4], A)
end
# Realize by Zygote
function get_norm_dA1(env::EnvTensor, phi, phid)
    A = get_A(phi)
    Ad = get_Ad(phid)
    Cs = corner(env)
    Es = edge(env)

    f = _x -> contract_env(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], Es[2], Es[3], Es[4], A, _x)

    gradient(f, Ad)[1]
end


