function optim_gs(H, A0, filename::String; m = 10, g_tol=1e-6, iterations = 200)
    # fprint("$(@__DIR__)/config.toml")
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        fprint("load daufult config file")
    end
    print_cfg(cfg)

    optim_gs(H, A0, cfg; m = m, g_tol= g_tol, iterations = iterations)
end

function optim_gs(H, A0, cfg::Dict; m = 10, g_tol=1e-6, iterations = 200)
    energies = Float64[]
    gradnorms = Float64[]

    cached_x = nothing
    cached_y = nothing 
    cached_g = nothing

    gs_name = get_gs_name(cfg)

    ts0 = CTMTensors(cfg)

    function verbose(xk)
        if cached_y !== nothing && cached_g !== nothing 
            append!(energies, cached_y)
            append!(gradnorms, norm(cached_g))
        end
        fprint(" # ======================== #")
        fprint(" #      Step completed      #")
        fprint(" # ======================== #")
        [@printf(" Step %3d  E: %0.8f  |grad|: %0.8f \n", i, E, gradnorms[i]) for (i, E) in enumerate(energies)]
        jldsave(gs_name; A = xk.metadata["x"], xk = xk, energies = energies, gradnorms = gradnorms)
        fprint("save to $gs_name")
        if cfg["gc"]
            GC.gc()
        end
        return false
    end

    if cfg["resume"] == true && ispath(gs_name)
        energies = load(gs_name, "energies")
        gradnorms = load(gs_name, "gradnorms")
        A0 = load(gs_name, "A")
        xk = load(gs_name, "xk")
        fprint("Resuming existing simulation")
        verbose(xk)
    end

    function fg!(F,G,x)
        # x = get_symmetry(x)
        x = renormalize(x)

        if cached_g !== nothing && cached_x !== nothing && norm(x - cached_x) < 1e-14
            fprint("Restart to find x")
            if G !== nothing
                copy!(G, cached_g)
            end
            if F !== nothing
                return cached_y
            end
        end

        Cs, Es = init_ctm(x)
        ts = setproperties(ts0, Cs = Cs, Es = Es, A = x, Ad = conj(x))
        conv_fun(_x) = get_gs_energy(_x, H)[1]
        fprint("\n ---- Start to find fixed points -----")
        ts, _ = run_ctm(ts; conv_fun = conv_fun)
        # ts, _ = run_ctm(ts)
        fprint("---- End to find fixed points ----- \n")

        max_iter = ts.Params["max_iter"]
        ts.Params["max_iter"] = ts.Params["ad_max_iter"]

        y, back = Zygote.pullback(_x -> run_gs(ts, H, _x), x)

        ts.Params["max_iter"] = max_iter

        fprint("Finish autodiff")
        y = real(y)
        cached_x = x
        cached_y = y

        if G !== nothing
            g = back(1)[1]
            cached_g = g
            # @show g
            copy!(G, g)
        end
        if F !== nothing
            @printf("Gs_Energy: %.10g \n", y)
            return y
        end
    end

    # optimizer = L_BFGS_B(1024, 17)
    # res = optimizer(Optim.only_fg!(fg!), A0, m=20, factr=1e7, pgtol=1e-5, iprint=-1, maxfun=15000, maxiter=15000)

    res = optimize(Optim.only_fg!(fg!), A0, LBFGS(m=m), Optim.Options(x_tol = 0.0, f_tol = 0.0, g_tol=g_tol, callback = verbose, iterations = iterations, extended_trace = true))

    res
end

function run_gs(ts::CTMTensors, H, A)
    ts1 = setproperties(ts, A = A, Ad = conj(A))

    conv_fun(_x) = get_gs_energy(_x, H)[1]
    ts1, s = run_ctm(ts1, conv_fun = conv_fun)
    # ts, _ = run_ctm(ts)
    
    # ts, s = iPEPS.run_ctm(conv_ts, 50)
    gs_E, _ = get_gs_energy(ts1, H)

    # gs_E = sum(E) |> real
    # @printf("Gs_Energy: %.10g \n", sum(E))
    # fprint("E: $E, N: $N")
    gs_E
end

function get_e0_4x4(ts, H)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = iPEPS.get_dm4(C1, C2, C3, C4, E1, E2, E3, E4, ts.A, ts.A, ts.A, ts.A)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:4]), :)

    nB = tr(n_dm)
    n_dm = n_dm./nB
    e0 = tr(H*n_dm)

    # iPEPS.fprint("E0: $e0    nB: $(nB)")

    e0, nB
end