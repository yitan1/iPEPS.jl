function optim_GS(H, A0)
    energies = Float64[]
    gradnorms = Float64[]

    cached_x = nothing
    cached_y = nothing 
    cached_g = nothing

    # println("$(@__DIR__)/config.toml")
    if ispath("config.toml")
        cfg = TOML.parsefile("config.toml")
        println("load custom config file")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        println("load daufult config file")
    end
    display(cfg)
    gs_name = get_gs_name(cfg)

    function verbose(xk)
        if cached_y !== nothing && cached_g !== nothing 
            append!(energies, cached_y)
            append!(gradnorms, norm(cached_g))
        end
        println(" # ======================== #")
        println(" #      Step completed      #")
        println(" # ======================== #")
        [@printf(" Step %3d  E: %0.8f  |grad|: %0.8f \n", i, E, gradnorms[i]) for (i, E) in enumerate(energies)]
        jldsave(gs_name; A = xk.metadata["x"], xk = xk, energies = energies, gradnorms = gradnorms)

        return false
    end

    if cfg["resume"] == true && ispath(gs_name)
        energies = load(gs_name, "energies")
        gradnorms = load(gs_name, "gradnorms")
        A0 = load(gs_name, "A")
        xk = load(gs_name, "xk")
        println("Resuming existing simulation")
        verbose(xk)
    end

    function fg!(F,G,x)
        x = renormalize(x)

        if cached_g !== nothing && cached_x !== nothing && norm(x - cached_x) < 1e-14
            println("Restart to find x")
            if G !== nothing
                copy!(G, cached_g)
            end
            if F !== nothing
                return cached_y
            end
        end

        ts0 = CTMTensors(x)
        conv_fun(_x) = get_gs_energy(_x, H)
        println("\n ---- Start to find fixed points -----")
        ts0, _ = run_ctm(ts0; conv_fun = conv_fun)
        println("---- End to find fixed points ----- \n")
        f(_x) = run_gs(ts0, H, _x) 
        y, back = Zygote.pullback(f, x)

        println("Finish autodiff")
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

    res = optimize(Optim.only_fg!(fg!), A0, LBFGS(m=20), inplace = false, Optim.Options(g_tol=1e-6, callback = verbose, iterations = 100, extended_trace = true))

    res
end

function run_gs(ts0::CTMTensors, H, A)
    ts0 = setproperties(ts0, A = A, Ad = conj(A))

    conv_fun(_x) = get_gs_energy(_x, H)
    ts, s = run_ctm(ts0, conv_fun = conv_fun)
    
    # ts, s = iPEPS.run_ctm(conv_ts, 50)
    gs_E = get_gs_energy(ts, H)

    # gs_E = sum(E) |> real
    # @printf("Gs_Energy: %.10g \n", sum(E))
    # println("E: $E, N: $N")
    gs_E
end