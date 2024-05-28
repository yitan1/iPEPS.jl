function hb_xx_k(Jx=1, Jy=1; Jz = 1, K = 0.1)
    hv = Jy * tout(tout(SI, sigmay), tout(sigmay, SI)) .+ (tout(tout(sigmax, sigmax), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmax, sigmax))) / 2 / 2 .+ K * tout(tout(sigmax, sigmaz), tout(sigmay, SI)) .+ K * tout(tout(SI, sigmay), tout(sigmaz, sigmax)) 
    hh = Jz * tout(tout(SI, sigmaz), tout(sigmaz, SI)) .+ (tout(tout(sigmax, sigmax), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmax, sigmax))) / 2 / 2 .+ K * tout(tout(sigmax, sigmay), tout(sigmaz, SI)) .+ K * tout(tout(SI, sigmaz), tout(sigmay, sigmax))
    
    [-hh, -hv]
end  


function hb_k_h4()
    II = tout(SI, SI) 
    Iy = tout(SI, sigmay)
    yI = tout(sigmay, SI)
    Iz = tout(SI, sigmaz)
    zI = tout(sigmaz, SI)
    xx = tout(sigmax, sigmax)

    xz = tout(sigmax, sigmaz)
    zx = tout(sigmaz, sigmax)
    xy = tout(sigmax, sigmay)
    yx = tout(sigmay, sigmax)

    h1 = tout(tout(Iy, yI) , tout(II, II)) .+ K*tout(tout(xz, yI) , tout(II, II)) .+ K*tout(tout(Iy, zx) , tout(II, II))
    h2 = tout(tout(II, II), tout(Iy, yI)) .+ K*tout(tout(II, II), tout(xz, yI)) .+ K*tout(tout(II, II), tout(Iy, zx))
    h3 = tout(tout(Iz, II), tout(zI, II)) .+ K*tout(tout(xy, II), tout(zI, II)) .+ K*tout(tout(Iz, II), tout(yx, II))
    h4 = tout(tout(II, Iz), tout(II, zI)) .+ K*tout(tout(II, xy), tout(II, zI)) .+ K*tout(tout(II, Iz), tout(II, yx))
    h5 = tout(tout(II, xx), tout(II, II)) 
    h6 = tout(tout(II, II), tout(xx, II))
    # h7 = tout(tout(xx, II), tout(II, II))
    # h8 = tout(tout(II, II), tout(II, xx))
    h = -h1 .- h3 .- h5 .- h2 .- h4 .- h6
    h./2
end