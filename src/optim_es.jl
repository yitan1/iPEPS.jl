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
    idx = sortperm(real.(ev_N))[end:-1:1]
    es = es[ixs]
    vecs = vecs[:,ixs]

    es, vecs
end

function optim_es(H, px, py, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)

    optim_es(H, px, py, cfg)
end

function optim_es(H, px, py, cfg::Dict)
    gs_name = get_gs_name(cfg)
    if ispath(gs_name)
        A = load(gs_name, "A")
        fprint("load existed ground state at $(gs_name)")
    else
        fprint("File of ground state is NOT existed, please run 'optim_gs(...) ' ")
        return nothing
    end

    A = renormalize(A)
    ts0 = CTMTensors(A, cfg)
    conv_fun(_x) = get_gs_energy(_x, H)[1]
    ts, _ = run_ctm(ts0, conv_fun = conv_fun)
    
    optim_es(ts, H, px, py)
end
function optim_es(ts0::CTMTensors, H, px, py)
    ## normalize gs
    ts = normalize_gs(ts0)

    H = substract_gs_energy(ts, H)

    #basis
    basis_name = get_basis_name(ts.Params)
    # post = ".jld2"
    if ispath(basis_name)
        basis = load(basis_name, "basis")
        fprint("The basis has existed, skip calculation")
    else
        basis = get_tangent_basis(ts)
        jldsave(basis_name; basis = basis)
        fprint("Saved the basis to $(basis_name)")
    end

    basis_dim = size(basis, 2)
    effH = zeros(ComplexF64, basis_dim, basis_dim)
    effN = zeros(ComplexF64, basis_dim, basis_dim)

    ts.Params["px"] = px*pi
    ts.Params["px"] = py*pi

    for i = 1:basis_dim
        fprint(" \n Starting simulation of basis vector $(i)/$(basis_dim)")
        gH, gN = get_es_grad(ts, H, basis[:,i])
        gH = conj(gH)
        effH[:, i] = basis' * gH
        effN[:, i] = basis' * gN
        fprint("Finish basis vector of $(i)/$(basis_dim)")
    end
    
    es_name = get_es_name(ts.Params, px, py)
    jldsave(es_name; effH = effH, effN = effN)
    fprint("Saved (effH, effN) to $(es_name)")

    effH, effN
end

function get_es_grad(ts0::CTMTensors, H, Bi)
    B = reshape(Bi, size(ts0.A))
    Cs, Es = init_ctm(ts0.A, ts0.Ad)

    ts1 = setproperties(ts0, Cs = Cs, Es = Es, B = B, Bd = conj(B))

    fprint("\n ---- Start to find fixed points -----")
    ts1, _ = run_ctm(ts1)
    fprint("---- End to find fixed points ----- \n")
    f(_x) = run_es(ts1, H, _x) 
    (y, ts), back = Zygote.pullback(f, B)
    gradH = back((1, nothing))[1]
    all_norm, gradN = get_all_norm(ts)
    fprint("Energy: $y \nNorm: $(all_norm[1][1]), $(all_norm[4][1]) ")
    
    gradH[:], gradN[:]
end

function run_es(ts0::CTMTensors, H, B)

    ts1 = setproperties(ts0, B = B, Bd = conj(B))

    ts, s = run_ctm(ts1)
    
    # ts, s = iPEPS.run_ctm(conv_ts, 50)
    E = get_es_energy(ts, H)

    # @printf("Gs_Energy: %.10g \n", sum(E))
    # fprint("E: $E, N: $N")

    E, ts
end

function normalize_gs(ts::CTMTensors)
    nrm = get_gs_norm(ts)
    A1 = ts.A ./ sqrt(abs(nrm))
    ts1 = setproperties(ts, A = A1, Ad = conj(A1))

    ts1
end

function substract_gs_energy(ts::CTMTensors, H)
    gs_E, _ = get_gs_energy(ts, H)
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
