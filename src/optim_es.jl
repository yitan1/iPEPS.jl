function make_es_path(step = 0.1)
    n = Int(1/step)
    # (1, 0) -> (1, 1)
    kx1 = ones(n-1)  # [1, 1]
    ky1 = collect( (0+step):step:(1-step) ) # (0, 1)
    # [1, 1] -> [0, 0]
    kx2 = collect(1:-step:0) # [1, 0]
    ky2 = collect(1:-step:0) # [1, 0]
    # (0,0) -> (1,0)
    kx3 = collect((0+step):step:(1-step)) 
    ky3 = zeros(n-1) 
    # [1,0] -> (0.5, 0.5)
    kx4 = collect(1:-step:(0.5 + step)) 
    ky4 = collect(0:step:(0.5-step) ) 

    kx = [kx1; kx2; kx3; kx4]
    ky = [ky1; ky2; ky3; ky4]

    kx, ky
end

function evaluate_es(px, py, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load dafault config file")
    end
    print_cfg(cfg)

    evaluate_es(px, py, cfg)
end
function evaluate_es(px, py, cfg::Dict)
    es_name = get_es_name(cfg, px, py)
    effH = load(es_name, "effH")
    effN = load(es_name, "effN")
    fprint("load H and N at $es_name")

    nrmB_cut = get(cfg, "nrmB_cut", 1e-3)

    H = (effH + effH') /2 
    N = (effN + effN') /2
    ev_N, P = eigen(N)
    idx = sortperm(real.(ev_N))[end:-1:1]
    ev_N = ev_N[idx]
    selected = (ev_N/maximum(ev_N) ) .> nrmB_cut
    P = P[:,idx]
    P = P[:,selected]
    N2 = P' * N * P
    H2 = P' * H * P
    H2 = (H2 + H2') /2 
    N2 = (N2 + N2') /2
    es, vecs = eigen(H2,N2)
    ixs = sortperm(real.(es))
    es = es[ixs]
    vecs = vecs[:,ixs]

    es, vecs
end

function prepare_basis(H, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)

    gs_name = get_gs_name(cfg)
    if ispath(gs_name)
        A = load(gs_name, "A")
        fprint("load existed ground state at $(gs_name)")
    else
        fprint("File of ground state is NOT existed, please run 'optim_gs(...) ' ")
        return nothing
    end

    A = renormalize(A)
    ts = CTMTensors(A, cfg)

    cfg_back = deepcopy(cfg)
    ts.Params["max_iter"] = 30
    ts.Params["rg_tol"] = 1e-12
    conv_fun(_x) = get_gs_energy(_x, H)[1]
    ts, _ = run_ctm(ts, conv_fun = conv_fun)

    ts.Params["max_iter"] = 30
    ts.Params["rg_tol"] = 1e-12

    ts = setproperties(ts, Params = cfg_back)

    ## normalize gs
    ts = normalize_gs(ts)

    H = substract_gs_energy(ts, H)

    #basis
    basis_name = get_basis_name(ts.Params)
    # post = ".jld2"
    if ispath(basis_name) && ts.Params["basis"]
        basis = load(basis_name, "basis")
        fprint("The basis has existed, skip calculation")
    else
        basis = get_tangent_basis(ts)
        jldsave(basis_name; basis = basis, ts = ts, H = H)
        fprint("Saved the basis, ts, H to $(basis_name)")
        # error()
    end
end

function optim_es(px, py, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)

    optim_es(px, py, cfg)
end

function optim_es(px, py, cfg::Dict)

    basis_name = get_basis_name(cfg)
    basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")
    H = load(basis_name, "H")
    fprint("load basis, ts, H in $basis_name")

    basis = complex(basis) # ！！！！ convert Complex
    basis_dim = size(basis, 2) 
    effH = zeros(ComplexF64, basis_dim, basis_dim)
    effN = zeros(ComplexF64, basis_dim, basis_dim)

    ts.Params["px"] = convert(eltype(ts.A), px*pi)
    ts.Params["py"] = convert(eltype(ts.A), py*pi)

    for i = 1:basis_dim
        fprint(" \n Starting simulation of basis vector $(i)/$(basis_dim)")
        # B = load("simulation/ising_default_D2_X30/gs.jld2", "A") .|> ComplexF64
        # B = reshape(B, 2, 2, 2,2, 2)
        # B = permutedims(B, (5, 4, 1,2 ,3))
        # B = permutedims(B, (5, 4, 3, 2 ,1))
        # display(B[:])
        # gH, gN = get_es_grad(ts, H, B)

        gH, gN = get_es_grad(ts, H, basis[:,i])
        # gN = reshape(gN, 2, 2, 2,2, 2)
        # gN = permutedims(gN, (5, 4, 1,2 ,3))
        # gN = permutedims(gN, (5, 4, 3,2 ,1))
        # display(gN[:])
        # @show i 
        # error()
        effH[:, i] = transpose(basis) * gH
        effN[:, i] = transpose(basis) * gN
        fprint("Finish basis vector of $(i)/$(basis_dim)")
    end
    
    es_name = get_es_name(ts.Params, px, py)
    jldsave(es_name; effH = effH, effN = effN)
    fprint("Saved (effH, effN) to $(es_name)")

    effH, effN
end

function optim_es1(ts::CTMTensors, H, px, py)
        ## normalize gs
        ts = normalize_gs(ts)

        H = substract_gs_energy(ts, H)
    
        #basis
        basis_name = get_basis_name(ts.Params)
        # post = ".jld2"
        if ispath(basis_name) && ts.Params["basis"]
            basis = load(basis_name, "basis")
            fprint("The basis has existed, skip calculation")
        else
            basis = get_tangent_basis(ts)
            jldsave(basis_name; basis = basis)
            fprint("Saved the basis to $(basis_name)")
        end
    
        basis = complex(basis) # ！！！！ convert Complex
        basis_dim = size(basis, 2) 
        effH = zeros(ComplexF64, basis_dim, basis_dim)
        effN = zeros(ComplexF64, basis_dim, basis_dim)
    
        ts.Params["px"] = convert(eltype(ts.A), px*pi)
        ts.Params["py"] = convert(eltype(ts.A), py*pi)
    
        f = x -> get_es_grad(ts, H, x)
        
        vals , vec = geneigsolve(f, 32, 3, :LM, ishermitian = true)
end

function get_es_grad(ts::CTMTensors, H, Bi)
    B = reshape(Bi, size(ts.A))
    Cs, Es = init_ctm(ts.A, ts.Ad)

    ts = setproperties(ts, Cs = Cs, Es = Es, B = B, Bd = conj(B))

    fprint("\n ---- Start to find fixed points -----")
    ts, _ = run_ctm(ts)
    fprint("---- End to find fixed points ----- \n")
    # f(_x) = run_es(ts1, H, _x) 
    (y, ts1), back = Zygote.pullback(x -> run_es(ts, H, x), B)
    gradH = back((1, nothing))[1]
    Nb, gradN = get_all_norm(ts1)
    fprint("Energy: $y \nNormB: $(Nb) ")
    
    gradH[:], gradN[:] ## conj???????
end

function run_es(ts::CTMTensors, H, B)

    ts = setproperties(ts, B = B, Bd = conj(B))

    ts, s = run_ctm(ts)
    
    # ts, s = iPEPS.run_ctm(conv_ts, 50)
    E = get_es_energy(ts, H)

    # Nb, gradN = get_all_norm(ts)
    # @printf("Gs_Energy: %.10g \n", sum(E))
    # fprint("Energy: $E \nNormB: $(Nb) ")

    E, ts
end

function normalize_gs(ts::CTMTensors)
    nrm = get_gs_norm(ts)
    fprint("Gs Norm: $nrm")
    A1 = ts.A ./ sqrt(abs(nrm))
    ts = setproperties(ts, A = A1, Ad = conj(A1))

    nrm = get_gs_norm(ts)
    fprint("Gs Norm: $nrm")

    ts
end

function substract_gs_energy(ts::CTMTensors, H)
    gs_E, _ = get_gs_energy(ts, H)
    gs_E = gs_E / 2

    newH = similar(H)
    newH[1] = H[1] .- gs_E* Matrix{eltype(H[1])}(I, size(H[1]))
    newH[2] = H[2] .- gs_E* Matrix{eltype(H[2])}(I, size(H[2]))
    fprint("Substracting $gs_E from Hamiltonian")

    # @show get_gs_energy(ts, newH)[1]

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

    # bs = load("one_D2_X30_base.jld2", "bs")
    # display(bs[:,1])
    # for i = 1:31
    #     b1 = reshape(bs[:,i], 2,2,2,2,2)
    #     b1 = permutedims(b1, (5,4,3,2,1)) 
    #     bs[:,i] = permutedims(b1, (3,4,5,2,1))[:] 
    # end
    # bs

    basis
end
