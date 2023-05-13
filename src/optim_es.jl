function optim_es(H, B)

    ts0 = CTMTensors(x, x);
    conv_fun(_x) = get_gs_energy(H, _x)
    println("\n ---- Start to find fixed points -----")
    ts0, _ = run_ctm(ts0, chi; conv_fun = conv_fun);
    println("---- End to find fixed points ----- \n")
    f(_x) = run_energy(H, ts0, chi, _x) 
    y, back = Zygote.pullback(f, x)

end

function run_es(H, ts0, chi, A)
    # ts0 = iPEPS.CTMTensors(A,A)
    ts0 = setproperties(ts0, A = A, Ad = conj(A))

    conv_fun(_x) = get_gs_energy(H, _x)
    ts, s = run_ctm(ts0, chi, conv_fun = conv_fun)
    
    # ts, s = iPEPS.run_ctm(conv_ts, 50)
    E, N = get_energy(H, ts)

    gs_E = sum(E) |> real
    # @printf("Gs_Energy: %.10g \n", sum(E))
    # println("E: $E, N: $N")

    gs_E
end