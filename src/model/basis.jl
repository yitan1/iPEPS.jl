const Sx = Float64[0 1; 1 0] / 2
const Sy = ComplexF64[0 -1im; 1im 0] / 2
const Sz = Float64[1 0; 0 -1] / 2
const SI = Float64[1 0; 0 1]
# sp = [0 1; 0 0]
# sm = [0 0; 1 0]
const sigmax = Float64[0 1; 1 0]
const sigmay = ComplexF64[0 -1im; 1im 0]
const sigmaz = Float64[1 0; 0 -1]
const sI = Float64[1 0; 0 1]

# Sx = [0 sqrt(3) 0 0; sqrt(3) 0 2 0; 0 2 0 sqrt(3); 0 0 sqrt(3) 0] / 2
# Sy = [0 -1im*sqrt(3) 0 0; 1im*sqrt(3) 0 -1im*2 0; 0 1im*2 0 -1im*sqrt(3); 0 0 1im*sqrt(3) 0] / 2
# Sz = [3 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -3] / 2

function tout(a, b)
    c = tcon([a, b], [[-1, -3], [-2, -4]])
    dim = size(c)
    c = reshape(c, dim[1] * dim[2], dim[3] * dim[4])
    return c
end

function ising(h = 2.0)
    # H = -tout(Sx, Sx)*2 .- h * tout(Sz, SI) / 2  .- h * tout(SI, Sz) / 2 
    H = -tout(sigmax, sigmax) .- h * tout(sigmaz, SI) / 2 / 2 .- h * tout(SI, sigmaz) / 2 / 2
    # H = -tout(Sz, Sz) .+ h * tout(Sx, SI) / 2 / 2 .+ h * tout(SI, Sx) / 2 / 2

    [H, H]
end

function heisenberg(Jz=1)
    H = Jz * tout(Sz, Sz) - tout(Sx, Sx) - tout(Sy, Sy)

    [H, H]
end

function ising_h4(h = 2.5)
    II = tout_site(SI, SI)
    Ix = tout_site(SI, sigmax)
    xI = tout_site(sigmax, SI)
    zI = h * tout_site(sigmaz, SI)
    xx = tout_site(sigmax, sigmax)

    h12 = tout_site(xx, II)
    h23 = tout_site(Ix, xI)
    h34 = tout_site(II, xx)
    h41 = tout_site(xI, Ix)
    hz1 = tout_site(zI, II)
    hz2 = tout_site(II, zI)

    h = -h12 .- h23 .- h34 .- h41 .- hz1 .- hz2
    d = size(SI, 1)
    h = reshape(h, d*d, d*d, d*d, d*d)

    h ./ 2
end