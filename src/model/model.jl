




function compute_gs4(ts, H)
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4, ts.A, ts.A, ts.A, ts.A)
    n_dm = reshape(n_dm, prod(size(n_dm)[1:4]), :)

    N = tr(n_dm)
    n_dm = n_dm./N
    e0 = tr(H*n_dm)

    iPEPS.fprint("E0: $e0    N: $(N)")

    e0, N
end

