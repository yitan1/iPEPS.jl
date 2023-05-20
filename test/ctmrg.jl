function test()
    H = honeycomb(1,1)
    cfg = TOML.parsefile("config.toml")
    A = load("simulation/hb_g11_D4_X100/gs.jld2", "A")
    chi = 100

    A = iPEPS.renormalize(A)
    ts0 = iPEPS.CTMTensors(A, cfg)
    ts, _ = iPEPS.run_ctm(ts0)

    B = randn(size(A))

    Cs, Es = iPEPS.init_ctm(ts.A, ts.Ad)

    ts1 = setproperties(ts, Cs = Cs, Es = Es, B = B, Bd = conj(B))

    @time ts1, _ = iPEPS.rg_step(ts1, chi)
    @time ts1, _ = iPEPS.rg_step(ts1, chi)
    @time ts1, _ = iPEPS.rg_step(ts1, chi)
    @time ts1, _ = iPEPS.rg_step(ts1, chi)
    iPEPS.fprint("3")

    @time ts1, s = iPEPS.left_rg(ts1, chi)
    @time ts1, _ = iPEPS.right_rg(ts1, chi)
    @time ts1, _ = iPEPS.top_rg(ts1, chi)
    @time ts1, _ = iPEPS.bottom_rg(ts1, chi)
    iPEPS.fprint("4")

    @time P, Pd, s = iPEPS.get_projector_left(ts1, chi)

    @time newC1, newE4, newC4 = iPEPS.proj_left(ts1, P, Pd)
    iPEPS.fprint("5")

    @time A    = iPEPS.get_all_A(ts1)
    @time Ad   = iPEPS.get_all_Ad(ts1)
    @time C1, C2, C3, C4 = iPEPS.get_all_Cs(ts1)
    @time E1, E2, E3, E4 = iPEPS.get_all_Es(ts1)
    iPEPS.fprint("6")

    newC1 = begin
        @time C1E1 = iPEPS.tcon([C1, E1], [[-2, 1], [1, -1, -3, -4]])
        @time CEP = iPEPS.wrap_reshape(C1E1, size(C1E1, 1), :) * P
        @time iPEPS.wrap_permutedims(CEP, (2,1))
    end
    iPEPS.fprint("7")


    newC4 = begin
        @time C4E3 = iPEPS.tcon([C4, E3], [[-1, 1], [1, -4, -2, -3]])
        @time Pd * iPEPS.wrap_reshape(C4E3, :, size(C4E3, 4))
    end
    iPEPS.fprint("8")

    newE4 = begin
        @time E4A = iPEPS.tcon([E4, A], [[-1, -3, 1, -6], [-2, 1, -4, -5, -7]])
        @show size(E4), size(A)
        @time EAAd = iPEPS.tcon([E4A, Ad], [[-1, -2, -4, -5, -7, 1, 2], [-3, 1, -6, -8, 2]])
        @show size(E4A), size(Ad)
        @time EAAd = iPEPS.wrap_reshape(EAAd, prod(size(EAAd)[1:3]), prod(size(EAAd)[4:6]), size(EAAd,7), :)
        @time EPd = iPEPS.tcon([EAAd, Pd], [[1, -2, -3, -4], [-1, 1]])
        @show size(EAAd), size(Pd)
        @time iPEPS.tcon([EPd, P], [[-1, 1, -3, -4], [1, -2]])
    end
    iPEPS.fprint("9")
    
end