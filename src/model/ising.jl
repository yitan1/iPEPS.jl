function ising(h)
    # H = -tout(Sx, Sx)*2 .- h * tout(Sz, SI) / 2  .- h * tout(SI, Sz) / 2 
    H = -tout(sigmax, sigmax) .- h * tout(sigmaz, SI) / 2 / 2 .- h * tout(SI, sigmaz) / 2 / 2
    # H = -tout(Sz, Sz) .+ h * tout(Sx, SI) / 2 / 2 .+ h * tout(SI, Sx) / 2 / 2

    [H, H]
end