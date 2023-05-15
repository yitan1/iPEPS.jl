function optim_es(ts::CTMTensors, H, px, py)
    ## normalize gs
    ts = normalize_gs(ts)

    H = substract_gs_energy(ts, H)

    basis = get_tangent_basis(ts)

    #basis
    out_prefix = get(ts.Params, "out_prefix", "none")
    jldsave("$(out_prefix)_basis.jld2"; basis = basis)
    println("saved the basis to $(out_prefix)_basis.jld2")

    basis_dim = size(basis, 2)
    H = zeros(ComplexF64, basis_dim, basis_dim)
    N = zeros(ComplexF64, basis_dim, basis_dim)

    for i = 1:basis_dim
        println("Starting simulation of basis vector $(i)/$(basis_dim)")
        gH, gN = get_es_grad(H, ts.A, basis[:,i], px, py)
        H[:, i] = basis' * gH
        N[:, i] = basis' * gN
        println("Finish basis vector of $(i)/$(basis_dim)")
    end
    
    jldsave("$(out_prefix)_es_$(px)_$(py).jld2"; H = H, N = N)
    println("saved (H, N) to $(out_prefix)_es_$(px)_$(py).jld2")

    H, N
end

function get_es_grad(H, A, Bi, px, py)
    B = reshape(Bi, size(A))
    ts0 = CTMTensors(A)
    ts0.Params["px"] = px
    ts0.Params["px"] = py

    ts1 = setproperties(ts0, B = B, Bd = conj(B))

    println("\n ---- Start to find fixed points -----")
    ts1, _ = run_ctm(ts1)
    println("---- End to find fixed points ----- \n")
    f(_x) = run_es(ts1, H, _x) 
    (y, ts), back = Zygote.pullback(f, B)
    gradH = back((1, nothing))[1]
    all_norm, gradN = get_all_norm(ts)
    println("Energy: $y \nNorm: $(all_norm[1][1]), $(all_norm[4][1]) ")
    
    gradH[:], gradN[:]
end

function run_es(ts0::CTMTensors, H, B)

    ts0 = setproperties(ts0, B = B, Bd = conj(B))

    ts, s = run_ctm(ts0)
    
    # ts, s = iPEPS.run_ctm(conv_ts, 50)
    E = get_es_energy(ts, H)

    # @printf("Gs_Energy: %.10g \n", sum(E))
    # println("E: $E, N: $N")

    E, ts
end

function normalize_gs(ts::CTMTensors)
    nrm = get_gs_norm(ts)
    A1 = ts.A ./ sqrt(abs(nrm))
    ts1 = setproperties(ts, A = A1, Ad = conj(A1))

    ts1
end

function substract_gs_energy(ts::CTMTensors, H)
    gs_E = get_gs_energy(ts, H)
    gs_E = gs_E / 2

    newH = similar(H)
    newH[1] = H[1] .- gs_E* Matrix{eltype(H[1])}(I, size(H[1]))
    newH[2] = H[2] .- gs_E* Matrix{eltype(H[2])}(I, size(H[2]))

    newH
end

function get_tangent_basis(ts::CTMTensors)
    # A = ts.A
    Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)
    ndm_Ad = tcon([n_dm, Ad], [[-1,-2,-3,-4,1,2,3,4], [1,2,3,4,-5]])

    ndm_Ad = reshape(ndm_Ad, 1, :)
    basis = nullspace(ndm_Ad)   #(dD^4, dD^4-1) 
    
    basis
end

