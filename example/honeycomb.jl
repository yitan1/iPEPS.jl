let 
    using iPEPS
    using MKL
    using TOML

    H = iPEPS.honeycomb(1, 1);

    A = iPEPS.init_hb_gs() |> real;

    ts0 = iPEPS.CTMTensors(A)

    gs_name = get_gs_name(ts0.Params)

    jldsave(gs_name; cfg = cfg)
    println("It's worked")

    # iPEPS.optim_GS(H, A)
end


