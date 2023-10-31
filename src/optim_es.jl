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
    if nrmB_cut isa Int
        selected = ev_N .> ev_N[nrmB_cut+1]
    else
        selected = (ev_N/maximum(ev_N) ) .> nrmB_cut
    end
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

    # A = get_symmetry(A)
    A = renormalize(A) 
    ts = CTMTensors(A, cfg)

    cfg_back = deepcopy(cfg)
    ts.Params["max_iter"] = 2*cfg["max_iter"]
    ts.Params["rg_tol"] = cfg["rg_tol"]^2
    conv_fun(_x) = get_gs_energy(_x, H)[1]
    ts, _ = run_ctm(ts, conv_fun = conv_fun)

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

    if cfg["ad"]
        return optim_es(px, py, cfg)
    else
        return optim_es_noad(px, py, cfg)
    end
end

function optim_es(px, py, cfg::Dict)

    basis_name = get_basis_name(cfg)
    basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")
    ts = setproperties(ts, Params = cfg)
    H = load(basis_name, "H")
    fprint("load basis, ts, H in $basis_name")

    basis = complex(basis) # ！！！！ convert Complex
    basis_dim = size(basis, 2) 

    es_name = get_es_name(ts.Params, px, py)
    if  ts.Params["es_resume"] > 0 && ispath(es_name) 
        es_file = load(es_name)
        effH = es_file["effH"]
        effN = es_file["effN"]
        if haskey(es_file, "envB")
            envB = load(es_name, "envB")
            println("load envB")
        else
            envB = zeros(ComplexF64, size(basis))
            println("new envB")
        end
        fprint("load existed calculation , effH, effN in $es_name")
    else
        effH = zeros(ComplexF64, basis_dim, basis_dim)
        effN = zeros(ComplexF64, basis_dim, basis_dim)
        envB = zeros(ComplexF64, size(basis))
    end

    ts.Params["px"] = convert(eltype(ts.A), px*pi)
    ts.Params["py"] = convert(eltype(ts.A), py*pi)

    for i = 1:basis_dim
        if i < ts.Params["es_resume"]
            fprint(" Simulation of basis vector $(i)/$(basis_dim) existed, skip to next")
            continue
        end
        if ts.Params["es_num"] > 0 && i >= (ts.Params["es_resume"] + ts.Params["es_num"])
            fprint("\nUp to maximum simulation of basis vector $(i)/$(basis_dim) existed, end to calculation")
            break
        end
        fprint("\nStarting simulation of basis vector $(i)/$(basis_dim)")

        @time gH, gN = get_es_grad(ts, H, basis[:,i])
        envB[:, i] = gN
        effH[:, i] = transpose(conj(basis)) * gH / 2
        effN[:, i] = transpose(conj(basis)) * gN

        fprint("\nFinish basis vector of $(i)/$(basis_dim)")

        if ts.Params["save"]
            jldsave(es_name; effH = effH, effN = effN, envB = envB)
            fprint("Saved (effH, effN) and envB to $(es_name)")
        end

        if ts.Params["gc"]
            GC.gc()
        end
    end

    if ts.Params["save"]
        jldsave(es_name; effH = effH, effN = effN, envB = envB)
        fprint("Saved (effH, effN) and envB to $(es_name)")
    end

    if ts.Params["gc"]
        GC.gc()
    end

    effH, effN
end

function optim_es_noad(px, py, cfg)
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
        for j = 1:basis_dim
            Hij, Nij = get_es_element(ts, H, basis[:,i], basis[:,j])
            effH[j, i] = Hij
            effN[j, i] = Nij
        end
        fprint("Finish basis vector of $(i)/$(basis_dim)")
    end
    
    es_name = get_es_name(ts.Params, px, py)
    jldsave(es_name; effH = effH, effN = effN)
    fprint("Saved (effH, effN) to $(es_name)")

    effH, effN
end

function get_es_element(ts, H, Bi, Bj)
    Bi = reshape(Bi, size(ts.A))
    Bj = reshape(Bj, size(ts.A))

    Cs, Es = init_ctm(ts.A, ts.Ad)

    ts1 = setproperties(ts, Cs = Cs, Es = Es, B = Bj, Bd = conj(Bi))

    fprint("\n ---- Start to find fixed points -----")
    conv_fun(_x) = get_es_energy(_x, H) / get_all_norm(_x)[1]
    ts1, _ = run_ctm(ts1, conv_fun = conv_fun)
    fprint("---- End to find fixed points ----- \n")

    max_iter = ts1.Params["max_iter"]
    ts1.Params["max_iter"] = ts1.Params["ad_max_iter"]

    ts1, _ = run_ctm(ts1, conv_fun = conv_fun)
    y = get_es_energy(ts1, H)

    ts1.Params["max_iter"] = max_iter
    
    Nb, _ = get_all_norm(ts1)
    fprint("Energy: $y \nNormB: $(Nb) ")
    
    y, Nb
end 

function optim_es1(ts::CTMTensors, H, px, py)
        ## normalize gs
        ts1 = normalize_gs(ts)

        H = substract_gs_energy(ts1, H)
    
        #basis
        basis_name = get_basis_name(ts1.Params)
        # post = ".jld2"
        if ispath(basis_name) && ts1.Params["basis"]
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
    
        f = x -> get_es_grad(ts1, H, x)
        
        vals , vec = geneigsolve(f, 32, 3, :LM, ishermitian = true)
end

function get_es_grad(ts::CTMTensors, H, Bi)
    B = reshape(Bi, size(ts.A))
    Bd = conj(B)
    # Cs, Es = init_ctm(ts.A, ts.Ad)
    ts1 = setproperties(ts, B = B, Bd = Bd)
    # ts1 = setproperties(ts, Cs = Cs, Es = Es, B = B, Bd = Bd)

    fprint("\n ---- Start to find fixed points -----")
    conv_fun(_x) = get_es_energy(_x, H) / get_all_norm(_x)[1]
    ts1, _ = run_ctm(ts1, conv_fun = conv_fun)
    fprint("---- End to find fixed points ----- \n")
    # f(_x) = run_es(ts1, H, _x) 

    st_time = time()
    max_iter = ts1.Params["max_iter"]
    ts1.Params["max_iter"] = ts1.Params["ad_max_iter"]

    (y, ts1), back = Zygote.pullback(x -> run_es(ts1, H, x), B)

    ts1.Params["max_iter"] = max_iter
    
    gradH = back((1, nothing))[1]
    Nb, gradN = get_all_norm(ts1)
    ed_time = time()
    fprint("Energy: $y \nNormB: $(Nb) ; ad_time = $(ed_time - st_time)")
    
    gradH[:], gradN[:] ## conj???????
end

function run_es(ts::CTMTensors, H, B)

    ts1 = setproperties(ts, B = B, Bd = conj(B))

    conv_fun(_x) = get_es_energy(_x, H) / get_all_norm(_x)[1]
    ts1, s = run_ctm(ts1, conv_fun = conv_fun)
    
    # ts, s = iPEPS.run_ctm(conv_ts, 50)
    E = get_es_energy(ts1, H)

    # Nb, gradN = get_all_norm(ts)
    # @printf("Gs_Energy: %.10g \n", sum(E))
    # fprint("Energy: $E \nNormB: $(Nb) ")

    E, ts1
end

function normalize_gs(ts::CTMTensors)
    nrm = get_gs_norm(ts)
    fprint("Gs Norm: $nrm")
    A1 = ts.A ./ sqrt(abs(nrm))
    ts1 = setproperties(ts, A = A1, Ad = conj(A1))

    nrm = get_gs_norm(ts1)
    fprint("Gs Norm: $nrm")

    ts1
end

function substract_gs_energy(ts::CTMTensors, H)
    gs_E, _ = get_gs_energy(ts, H)
    gs_E = gs_E / 2 |> real

    newH = similar(H)
    newH[1] = H[1] .- gs_E* Matrix{eltype(H[1])}(I, size(H[1]))
    newH[2] = H[2] .- gs_E* Matrix{eltype(H[2])}(I, size(H[2]))
    fprint("Substracting $gs_E from Hamiltonian")

    # @show get_gs_energy(ts, newH)[1]

    newH
end

function get_tangent_basis(ts::CTMTensors)
    if ts.Params["basis_t"] == "unit"
        M = length(ts.A)
        basis = Matrix{ComplexF64}(I, M, M)
    elseif ts.Params["basis_t"] == "cut"
        basis = cut_basis(ts)
    elseif ts.Params["basis_t"] == "gauge"
        basis = get_gauge_basis(ts)
    else 
        # A = ts.A
        Ad = ts.Ad
        C1, C2, C3, C4 = ts.Cs
        E1, E2, E3, E4 = ts.Es

        n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)
        ndm_Ad = tcon([n_dm, Ad], [[-1,-2,-3,-4,1,2,3,4], [1,2,3,4,-5]])

        ndm_Ad = reshape(ndm_Ad, 1, :)
        basis = nullspace(ndm_Ad)   #(dD^4, dD^4-1) 
    end

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

function cut_basis(ts::CTMTensors)
    vs, vecs = diag_n_dm(ts)

    basis_cut = get(ts.Params, "basis_cut", 1e-3) 
    idx = sortperm(real.(vs))[end:-1:1]
    vs = vs[idx]
    display(vs)
    selected = vs./maximum(vs) .> basis_cut
    display(vs[selected])

    vecs = vecs[:, selected]

    d = size(ts.A, 5)
    basis = zeros(ComplexF64, size(vecs,1)*d, size(vecs,2)*d)
    v_phy = ones(d) |> diagm

    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es
    n_dm = iPEPS.get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4);
    ndm_Ad = iPEPS.tcon([n_dm, ts.Ad], [[-1,-2,-3,-4,1,2,3,4], [1,2,3,4,-5]]);
    nAA = transpose(ts.A[:])*ndm_Ad[:]

    for ind in axes(basis, 2)
        i = div(ind-1, d) + 1
        j = (ind-1) % d + 1
        bi = tcon([vecs[:,i], v_phy[:,j]], [[-1],[-2]])[:]
        
        basis[:, ind] = bi .- (transpose(bi) * ndm_Ad[:]) * ts.A[:]./nAA
    end

    # vs, vecs, n_dm
    basis
end

function get_gauge_basis(ts)
    D = size(ts.A, 1)/2 |> Int
    d = size(ts.A, 5)
    M = D^4 * d

    # act zero flux
    basis0 = zeros(ComplexF64, length(ts.A), M)
    for i = 1:M
        B0 = zeros(M)
        B0[i] = 1
        B0 = reshape(B0, (D,D,D,D,d))
        Bi = act_Q_op(B0; add = 0)
        basis0[:,i] = Bi[:]
    end

    # act two flux z
    basis1 = zeros(ComplexF64, length(ts.A), M)
    for i = 1:M
        B0 = zeros(M)
        B0[i] = 1
        B0 = reshape(B0, (D,D,D,D,d))
        Bi = act_Q_op(B0, add = 3)
        basis1[:,i] = Bi[:]
    end

    # act two flux x
    basis2 = zeros(ComplexF64, length(ts.A), M)
    for i = 1:M
        B0 = zeros(M)
        B0[i] = 1
        B0 = reshape(B0, (D,D,D,D,d))
        Bi = act_Q_op(B0, add = 1)
        basis2[:,i] = Bi[:]
    end

    # act two flux y
    basis3 = zeros(ComplexF64, length(ts.A), M)
    for i = 1:M
        B0 = zeros(M)
        B0[i] = 1
        B0 = reshape(B0, (D,D,D,D,d))
        Bi = act_Q_op(B0, add = 2)
        basis3[:,i] = Bi[:]
    end

    basis = [basis0 basis1 basis2 basis3]

    basis
end

function act_Q_op(A0; add = 0)
    Q_op = get_Q_op()
    if add == 1 # XX bond
        Q_op1 = tcon([Q_op, sigmax], [[1,-2,-3,-4,-5], [1,-1]])
    elseif add == 2 # YY bond
        Q_op1 = tcon([Q_op, sigmay], [[-1,1,-3,-4,-5], [1,-2]])
    elseif add == 3 # ZZ bond
        Q_op1 = tcon([Q_op, sigmaz], [[-1,-2,1,-4,-5], [1,-3]])
    elseif add == 0 
        Q_op1 = Q_op
    end
    QQ = tcon([Q_op1, Q_op], [[-1,-2,1,-5,-7], [-3,-4,1,-6,-8]])
    dim = size(QQ)
    QQ = reshape(QQ, dim[1],dim[2], dim[3],dim[4], dim[5]*dim[6], dim[7]*dim[8])

    A = tcon([QQ,A0], [[-1, -3, -5, -7, -9, 1], [-2, -4, -6, -8, 1]])
    D1 = size(A,1)
    D2 = size(A,2)
    A = reshape(A, D1*D2, D1*D2, D1*D2, D1*D2, size(A,9))

    A
end

function diag_n_dm(ts::CTMTensors)
    # A = ts.A
    # Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)

    n_dm = reshape(n_dm, prod(size(n_dm)[1:4]), :)
    n_dm = (n_dm + n_dm')/2

    vs, vecs = eigen(n_dm)

    vs, vecs
end

function diag_n_dm2(ts::CTMTensors)
    # A = ts.A
    # Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm2(C1, C2, C3, C4, E1, E2, E3, E4)

    n_dm = reshape(n_dm, prod(size(n_dm)[1:6]), :)
    n_dm = (n_dm + n_dm')/2

    vs, vecs = eigen(n_dm)

    vs, vecs
end

function diag_n_dm3(ts::CTMTensors)
    # A = ts.A
    # Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm3(C1, C2, C3, C4, E1, E2, E3, E4)

    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)
    n_dm = (n_dm + n_dm')/2

    vs, vecs = eigen(n_dm)

    vs, vecs
end

function diag_n_dm4(ts::CTMTensors)
    # A = ts.A
    # Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)

    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)
    n_dm = (n_dm + n_dm')/2

    vs, vecs = eigen(n_dm)

    vs, vecs
end


