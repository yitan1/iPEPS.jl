let 
    using iPEPS
    using MKL
    # using Optim
    # using LinearAlgebra

    H = iPEPS.honeycomb(1, 1);

    A = iPEPS.init_hb_gs() |> real;

    iPEPS.optim_GS(H, A)
end


