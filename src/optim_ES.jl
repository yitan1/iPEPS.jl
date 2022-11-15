
function optimize_ES(kx, ky, phi0::IPEPS, h; kwargs...) #TODO
    # Ad = conj(A)
    Bn = get_tangent_basis(phi0; kwargs)
    H = eff_Hamitonian(h, phi, Bn)
    N = eff_norm(phi, Bn)
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
Return a matrix(n,n)
"""
function eff_Hamitonian(h, phi, Bn)
    M = size(Bn,2)
    H = zeros(eltype(Bn), M, M) 
    for j in axes(H,2)
        Bj = Bn[j]
        Bdj = conj(B)
        
        hBj = gradient(_x -> get_energy(h, phi, Bj, _x), Bdj)[1]
        for i in axes(H,1)
            Bi = Bn[i]
            H[i,j] = Bi*hBj'
        end
    end
    H
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
function get_norm_dA1(env::EnvTensor, phi)
    A = get_A(phi)
    Ad = get_Ad(phi)
    Cs = corner(env)
    Es = edge(env)

    f = _x -> contract_env(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], Es[2], Es[3], Es[4], A, _x)

    gradient(f, Ad)[1]
end

###########
"""
    get_norm

```
C1 -- E1 -- C2
|     |     |     
E4 -- T  -- E2  = N
|     |     |  
C4 -- E3 -- C3
```
"""
function get_norm(env::EnvTensor)
    T = bulk(env)
    get_norm(env, T)
end

function get_norm(env::EnvTensor, T)
    Cs = corner(env)
    Es = edge(env)

    contract_env(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], Es[2], Es[3], Es[4], T)
end

