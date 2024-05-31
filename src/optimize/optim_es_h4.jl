function prepare_basis_h4(H, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end

    prepare_basis(H, cfg)
end

function prepare_basis_h4(H, cfg0::Dict)
    print_cfg(cfg0)

    cfg = deepcopy(cfg0)
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

    ts.Params["max_iter"] = 2 * cfg["max_iter"]
    ts.Params["rg_tol"] = cfg["rg_tol"]^2

    H_local = get_local_h(H)
    conv_fun(_x) = get_gs_energy_4x4(_x, H_local)

    ts, _ = run_ctm(ts, conv_fun=conv_fun)

    ts = setproperties(ts, Params=cfg0)

    ## normalize gs
    ts = normalize_gs(ts)

    H = substract_gs_energy_h4(ts, H)
    H = get_local_h(H) 

    #basis
    basis_name = get_basis_name(ts.Params)
    if ispath(basis_name) && ts.Params["basis"]
        basis = load(basis_name, "basis")
        fprint("The basis has existed, skip calculation")
    else
        basis = get_tangent_basis(ts)
        jldsave(basis_name; basis=basis, ts=ts, H=H)
        fprint("Saved the basis, ts, H to $(basis_name)")
    end
end

# function optim_es(px, py, filename::String)
#     if ispath(filename)
#         cfg = TOML.parsefile(filename)
#         fprint("load custom config file at $(filename)")
#     else
#         cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
#         fprint("load daufult config file")
#     end

#     if cfg["ad"]
#         return optim_es(px, py, cfg)
#     else
#         return optim_es_noad(px, py, cfg)
#     end
# end

function optim_es_h4(px, py, cfg::Dict)

    print_cfg(cfg)

    basis_name = get_basis_name(cfg)
    basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")
    ts = setproperties(ts, Params=cfg)
    H = load(basis_name, "H")
    fprint("load basis, ts, H in $basis_name")

    basis = complex(basis) # ！！！！ convert Complex
    basis_dim = size(basis, 2)

    es_name = get_es_name(ts.Params, px, py)
    if ts.Params["es_resume"] > 0 && ispath(es_name)
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

    ts.Params["px"] = convert(eltype(ts.A), px * pi)
    ts.Params["py"] = convert(eltype(ts.A), py * pi)

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

        @time gH, gN = get_es_grad_h4(ts, H, basis[:, i])
        envB[:, i] = gN
        effH[:, i] = transpose(conj(basis)) * gH / 2
        effN[:, i] = transpose(conj(basis)) * gN

        fprint("\nFinish basis vector of $(i)/$(basis_dim)")

        if ts.Params["save"]
            jldsave(es_name; effH=effH, effN=effN, envB=envB)
            fprint("Saved (effH, effN) and envB to $(es_name)")
        end

        if ts.Params["gc"]
            GC.gc()
        end
    end

    if ts.Params["save"]
        jldsave(es_name; effH=effH, effN=effN, envB=envB)
        fprint("Saved (effH, effN) and envB to $(es_name)")
    end

    if ts.Params["gc"]
        GC.gc()
    end

    effH, effN
end

function get_es_grad_h4(ts::CTMTensors, H, Bi)
    B = reshape(Bi, size(ts.A))
    Bd = conj(B)
    # Cs, Es = init_ctm(ts.A, ts.Ad)
    ts1 = setproperties(ts, B=B, Bd=Bd)
    # ts1 = setproperties(ts, Cs = Cs, Es = Es, B = B, Bd = Bd)

    fprint("\n ---- Start to find fixed points -----")
    conv_fun(_x) = get_es_energy_4x4(_x, H) / get_all_norm(_x)[1]
    ts1, _ = run_ctm(ts1, conv_fun=conv_fun)
    fprint("---- End to find fixed points ----- \n")
    # f(_x) = run_es(ts1, H, _x) 

    st_time = time()
    max_iter = ts1.Params["max_iter"]
    ts1.Params["max_iter"] = ts1.Params["ad_max_iter"]

    (y, ts1), back = Zygote.pullback(x -> run_es_h4(ts1, H, x), B)

    ts1.Params["max_iter"] = max_iter

    gradH = back((1, nothing))[1]
    Nb, gradN = get_all_norm(ts1)
    ed_time = time()
    fprint("Energy: $y \nNormB: $(Nb) ; ad_time = $(ed_time - st_time)")

    gradH[:], gradN[:] ## conj???????
end

function run_es_h4(ts::CTMTensors, H, B)
    ts1 = setproperties(ts, B=B, Bd=conj(B))

    conv_fun(_x) = get_es_energy_4x4(_x, H) / get_all_norm(_x)[1]
    ts1, s = run_ctm(ts1, conv_fun=conv_fun)

    E = get_es_energy_4x4(ts1, H)

    E, ts1
end

function substract_gs_energy_h4(ts::CTMTensors, H)
    gs_E = get_gs_energy_4x4(ts, get_local_h(H))

    d = sqrt(size(H, 1)) |> Int
    II = Matrix{eltype(H)}(I, d, d)
    hI = tout_site(tout_site(II, II), tout_site(II, II))
    hI = reshape(hI, d * d, d * d, d * d, d * d)

    newH = similar(H)
    newH = H - gs_E * hI
    fprint("Substracting $gs_E from Hamiltonian")

    newH
end