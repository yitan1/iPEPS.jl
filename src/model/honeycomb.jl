function honeycomb(Jx=1, Jy=1; Jz=1, dir="ZZ")
    if dir == "ZZ"
        hv = Jx * tout(tout(SI, sigmax), tout(sigmax, SI)) .+ (tout(tout(sigmaz, sigmaz), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmaz, sigmaz))) / 2 / 2 .|> real
        hh = Jy * tout(tout(SI, sigmay), tout(sigmay, SI)) .+ (tout(tout(sigmaz, sigmaz), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmaz, sigmaz))) / 2 / 2 .|> real
    elseif dir == "XX"
        hv = Jy * tout(tout(SI, sigmay), tout(sigmay, SI)) .+ (tout(tout(sigmax, sigmax), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmax, sigmax))) / 2 / 2 .|> real
        hh = Jz * tout(tout(SI, sigmaz), tout(sigmaz, SI)) .+ (tout(tout(sigmax, sigmax), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmax, sigmax))) / 2 / 2 .|> real
    elseif dir == "YY"
        hv = Jz * tout(tout(SI, sigmaz), tout(sigmaz, SI)) .+ (tout(tout(sigmay, sigmay), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmay, sigmay))) / 2 / 2 .|> real
        hh = Jx * tout(tout(SI, sigmax), tout(sigmax, SI)) .+ (tout(tout(sigmay, sigmay), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmay, sigmay))) / 2 / 2 .|> real
    end

    [-hh, -hv]
end

# XX
function honeycomb_h4()
    II = tout(SI, SI)
    Iy = tout(SI, sigmay)
    yI = tout(sigmay, SI)
    Iz = tout(SI, sigmaz)
    zI = tout(sigmaz, SI)
    xx = tout(sigmax, sigmax)

    h1 = tout(tout(Iy, yI), tout(II, II))
    h2 = tout(tout(II, II), tout(Iy, yI))
    h3 = tout(tout(Iz, II), tout(zI, II))
    h4 = tout(tout(II, Iz), tout(II, zI))
    h5 = tout(tout(II, xx), tout(II, II))
    h6 = tout(tout(II, II), tout(xx, II))
    # h7 = tout(tout(xx, II), tout(II, II))
    # h8 = tout(tout(II, II), tout(II, xx))
    h = -h1 .- h3 .- h5 .- h2 .- h4 .- h6
    h ./ 2
end