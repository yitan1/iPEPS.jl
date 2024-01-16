function optim_gs(H, A0, cfg::Dict; m = 10, kwargs...)
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

    opt = Optim.Options(;kwargs...)
    res = optimize(Optim.only_fg!(fg!), A0, LBFGS(m=m), opt)

    res
end