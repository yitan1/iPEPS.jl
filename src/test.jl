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