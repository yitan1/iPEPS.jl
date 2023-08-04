function test_es(chi, px, py, filename::String)
    if ispath(filename)
        cfg = TOML.parsefile(filename)
        iPEPS.fprint("load custom config file at $(filename)")
    else
        cfg = TOML.parsefile("$(@__DIR__)/default_config.toml")
        iPEPS.fprint("load dafault config file")
    end
    iPEPS.print_cfg(cfg)

    basis_name = iPEPS.get_basis_name(cfg)
    basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")
    H = load(basis_name, "H")
    iPEPS.fprint("load basis, ts, H in $basis_name")

    ts.Params["px"] = convert(eltype(ts.A), px*pi)
    ts.Params["py"] = convert(eltype(ts.A), py*pi)
    ts.Params["chi"] = chi

    Bi = basis[:, 1]
    B = reshape(Bi, size(ts.A))
    Cs, Es = iPEPS.init_ctm(ts.A, ts.Ad)

    ts = setproperties(ts, Cs = Cs, Es = Es, B = B, Bd = conj(B))

    iPEPS.fprint("\n ---- Start to find fixed points -----")
    ts, _ = iPEPS.run_ctm(ts)
    iPEPS.fprint("---- End to find fixed points ----- \n")
    # f(_x) = run_es(ts1, H, _x) 
    y, ts1 = iPEPS.run_es(ts, H, B)
    # gradH = back((1, nothing))[1]
    Nb, gradN = iPEPS.get_all_norm(ts1)
    iPEPS.fprint("Energy: $y \nNormB: $(Nb) ")
end

function renormalize1(A) 
    factor = maximum(abs.(A))
    A = A * (1 / factor)

    A, factor        
end

function renormalize1(A, factor) 
    A = A * (1 / factor)

    A   
end


function run_ctm(ts, ots; conv_fun = nothing)
    chi = get(ts.Params, "chi", 30)
    min_iter = get(ts.Params, "min_iter", 4)
    max_iter = get(ts.Params, "max_iter", 20)
    tol = get(ts.Params, "rg_tol", 1e-6)
    diffs = [1.0]
    old_conv = 1.0
    conv = 1.0
    

    for i = 1:max_iter
        st_time = time()
        ts, ots, s = rg_step(ts, ots, chi)
        e = get_gs_norm(ts)
        nrm = get_gs_norm(ots)
        N = (2*i + 3)^2
        println()
        @show nrm, e
        # @show e/nrm
        @show log(e/nrm)/N/0.0001

        ed_time = time()

        ctm_time = ed_time - st_time

        old_conv = conv
        if conv_fun !== nothing
            conv = Zygote.@ignore conv_fun(ts) 
        else
            conv = Zygote.@ignore s
        end

        if  length(conv) == length(old_conv)
            diff = Zygote.@ignore norm(conv .- old_conv)
            append!(diffs, diff)
        end

        if conv_fun !== nothing
            @printf("CTM step %3d, diff = %.4e, time = %.4f, obj = %.6f \n",i, diffs[end], ctm_time, conv)
        else
            @printf("CTM step %3d, diff = %.4e, time = %.4f \n", i, diffs[end], ctm_time)
        end

        if i >= min_iter && diffs[end] < tol 
            fprint("---------- CTM finished ---------")
            break
        end
        if i == max_iter && diffs[end] > tol 
            fprint("--------- Not Converged ----------")
        end

    end

    ts, ots, conv
end

function rg_step(ts, ots, chi)
    ts, ots, s = left_rg(ts, ots, chi)
    ts, ots, _ = right_rg(ts, ots, chi)
    ts, ots, _ = top_rg(ts, ots, chi)
    ts, ots, _ = bottom_rg(ts, ots, chi)

    ts, ots, s
end

function left_rg(ts, ots, chi)
    P, Pd, s = get_projector_left(ts, chi)
    oP, oPd, os = get_projector_left(ots, chi)
    newC1, newE4, newC4, onewC1, onewE4, onewC4 = proj_left(ts, P, Pd, ots, oP, oPd)

    ts = up_left(ts, newC1, newE4, newC4)
    ots = up_left(ots, onewC1, onewE4, onewC4)

    ts, ots, s
end

function right_rg(ts, ots, chi)
    P, Pd, s = get_projector_right(ts, chi)
    oP, oPd, os = get_projector_right(ots, chi)
    newC2, newE2, newC3, onewC2, onewE2, onewC3 = proj_right(ts, P, Pd, ots, oP, oPd)

    ts = up_right(ts, newC2, newE2, newC3)
    ots = up_right(ots, onewC2, onewE2, onewC3)

    ts, ots, s
end

function top_rg(ts, ots, chi)
    P, Pd, s = get_projector_top(ts, chi)
    oP, oPd, os = get_projector_top(ots, chi)
    newC1, newE1, newC2, onewC1, onewE1, onewC2 = proj_top(ts, P, Pd,  ots, oP, oPd)

    ts = up_top(ts, newC1, newE1, newC2)
    ots = up_top(ots, onewC1, onewE1, onewC2)
    ts, ots, s
end

function bottom_rg(ts, ots, chi)
    P, Pd, s = get_projector_bottom(ts, chi)
    oP, oPd, os = get_projector_bottom(ots, chi)
    newC4, newE3, newC3, onewC4, onewE3, onewC3 = proj_bottom(ts, P, Pd, ots, oP, oPd)

    ts = up_bottom(ts, newC4, newE3, newC3)
    ots = up_bottom(ots, onewC4, onewE3, onewC3)

    ts, ots, s
end

function proj_left(ts, P, Pd, ots, oP, oPd)
    newC1, newE4, newC4 = proj_left_unrenorm(ts, P, Pd)
    onewC1, onewE4, onewC4 = proj_left_unrenorm(ots, oP, oPd)
    
    newC1, f1 = renormalize1(newC1)
    onewC1 = renormalize1(onewC1, f1)

    newE4, f2 = renormalize1(newE4)
    onewE4 = renormalize1(onewE4, f2)

    newC4, f3 = renormalize1(newC4)
    onewC4 = renormalize1(onewC4, f3)

    newC1, newE4, newC4, onewC1, onewE4, onewC4
end

function proj_right(ts, P, Pd, ots, oP, oPd)
    newC2, newE2, newC3 = proj_right_unrenorm(ts, P, Pd)
    onewC2, onewE2, onewC3 = proj_right_unrenorm(ots, oP, oPd)
    
    newC2, f1 = renormalize1(newC2)
    onewC2 = renormalize1(onewC2, f1)

    newE2, f2 = renormalize1(newE2)
    onewE2 = renormalize1(onewE2, f2)

    newC3, f3 = renormalize1(newC3)
    onewC3 = renormalize1(onewC3, f3)

    newC2, newE2, newC3, onewC2, onewE2, onewC3
end

function proj_top(ts, P, Pd, ots, oP, oPd)
    newC1, newE1, newC2 = proj_top_unrenorm(ts, P, Pd)
    onewC1, onewE1, onewC2 = proj_top_unrenorm(ots, oP, oPd)
    
    newC1, f1 = renormalize1(newC1)
    onewC1 = renormalize1(onewC1, f1)

    newE1, f2 = renormalize1(newE1)
    onewE1 = renormalize1(onewE1, f2)

    newC2, f3 = renormalize1(newC2)
    onewC2 = renormalize1(onewC2, f3)

    newC1, newE1, newC2, onewC1, onewE1, onewC2
end

function proj_bottom(ts, P, Pd, ots, oP, oPd)
    newC4, newE3, newC3 = proj_bottom_unrenorm(ts, P, Pd)
    onewC4, onewE3, onewC3 = proj_bottom_unrenorm(ots, oP, oPd)
    
    newC4, f1 = renormalize1(newC4)
    onewC4 = renormalize1(onewC4, f1)

    newE3, f2 = renormalize1(newE3)
    onewE3 = renormalize1(onewE3, f2)

    newC3, f3 = renormalize1(newC3)
    onewC3 = renormalize1(onewC3, f3)

    newC4, newE3, newC3, onewC4, onewE3, onewC3
end

function finite_ctm(H, A, cfg)
    ts0 = CTMTensors(A, cfg)
    ts, _ = run_ctm(ts0)

    pepoN = init_pepo(H, 0.001)

    ots = get_ots(ts, pepoN)

    ots = setproperties(ts, A = ots.A, Ad = ots.Ad)

    ts, ots
end