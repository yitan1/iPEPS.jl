
function optimize_ES(phi::ExcIPEPS, h; kwargs...)
    # Ad = conj(A)
    Bn = get_tangent_basis(phi)
    H = eff_Hamitonian(h, phi, Bn)
    N = eff_norm(phi, Bn)
    energy, _ = eigsolve(H,N)
    
    energy
end

function get_tangent_basis(phi::ExcIPEPS)
    get_tangent_basis(phi::IPEPS)
end

function get_tangent_basis(phi::IPEPS)
    env = CE_matrix(phi)
    effB = normAB_dB(env)
    effB = reshape(effB, (1,:) )
    basis = nullspace(effB)   #(dD^4-1, dD^4)  

    basis
end

