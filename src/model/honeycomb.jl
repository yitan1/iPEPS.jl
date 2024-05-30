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

function tout_site(a, b)
    c = tcon([a, b], [[-1, -2], [-3, -4]])
    dim = size(c)
    c = reshape(c, dim[1] * dim[2], dim[3] * dim[4])
    return c
end

"""
1--- y ----2
|          |
x          x
|          | 
4----y-----3
"""
function hb_h4_ZZ()
    II = tout(SI, SI)
    Iy = tout(SI, sigmay)
    yI = tout(sigmay, SI)
    Ix = tout(SI, sigmax)
    xI = tout(sigmax, SI)
    zz = tout(sigmaz, sigmaz)
    
    hx1 = tout_site(tout_site(II, Ix), tout_site(xI, II))
    hx2 = tout_site(tout_site(Ix, II), tout_site(II, xI))
    hy1 = tout_site(tout_site(Iy, yI), tout_site(II, II))
    hy2 = tout_site(tout_site(II, II), tout_site(yI, Iy))
    hz1 = tout_site(tout_site(II, zz), tout_site(II, II))
    hz2 = tout_site(tout_site(II, II), tout_site(II, zz))

    h = -hx1 .- hx2 .- hy1 .- hy2 .- hz1 .- hz2
    d = size(II, 1)
    h = reshape(h, d*d, d*d, d*d, d*d)

    return h ./ 2
end
