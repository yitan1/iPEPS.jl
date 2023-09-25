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

function tout(a, b)
    c = tcon([a, b], [[-1, -3], [-2, -4]])
    dim = size(c)
    c = reshape(c, dim[1] * dim[2], dim[3] * dim[4])
    return c
end

function ising(h)
    # H = -tout(Sx, Sx)*2 .- h * tout(Sz, SI) / 2  .- h * tout(SI, Sz) / 2 
    H = -tout(sigmax, sigmax) .- h * tout(sigmaz, SI) / 2 /2 .- h * tout(SI, sigmaz) / 2/ 2
    # H = -tout(Sz, Sz) .+ h * tout(Sx, SI) / 2 / 2 .+ h * tout(SI, Sx) / 2 / 2

    [H, H]
end

function heisenberg(Jz=1)
    H = Jz * tout(Sz, Sz) - tout(Sx, Sx) - tout(Sy, Sy)

    [H, H]
end

function honeycomb(Jx=1, Jy=1)
    hv = Jx * tout(tout(SI, sigmax), tout(sigmax, SI)) .+ (tout(tout(sigmaz, sigmaz), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmaz, sigmaz))) / 2 / 2 .|> real
    hh = Jy * tout(tout(SI, sigmay), tout(sigmay, SI)) .+ (tout(tout(sigmaz, sigmaz), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmaz, sigmaz))) / 2 / 2 .|> real

    [-hh, -hv ]
end

function init_hb_gs(D=4; p1=0.24, p2=0.0)
    Q_op = zeros(ComplexF64, 2, 2, 2, 2, 2)
    Q_op[1, 1, 1, :, :] = SI
    Q_op[1, 2, 2, :, :] = Sx * 2
    Q_op[2, 1, 2, :, :] = Sy * 2
    Q_op[2, 2, 1, :, :] = Sz * 2
    ux, uy, uz = 1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)
    s111 = 1 / sqrt(2 + 2 * uz) * [1 + uz, ux + im * uy]

    T = tcon([Q_op, s111], [[-1, -2, -3, -4, 1], [1]])
    # A = tcon([T, T], [[1,-1,-2,-5], [1,-3,-4,-6]]) # XX
    # A = tcon([T, T], [[-1, 1, -4, -5], [-3, 1, -2, -6]]) # YY
    A = tcon([T, T], [[-1, -2, 1, -5], [-3, -4, 1, -6]]) # ZZ
    A = reshape(A, 2, 2, 2, 2, 4)

    if D == 4
        phi = p1 * pi
        theta = exp(-im * pi * p2)
        R_op = zeros(ComplexF64, 2, 2, 2, 2, 2)
        R_op[1, 1, 1, :, :] = SI .* cos(phi)
        R_op[2, 1, 1, :, :] = 2 * Sx * sin(phi) * theta
        R_op[1, 2, 1, :, :] = 2 * Sy * sin(phi) * theta
        R_op[1, 1, 2, :, :] = 2 * Sz * sin(phi) * theta

        RR = tcon([R_op, R_op], [[-1, -2, 1, -5, -7], [-3, -4, 1, -6, -8]])
        dRR = size(RR)
        RR = reshape(RR, dRR[1], dRR[2], dRR[3], dRR[4], dRR[5] * dRR[6], dRR[7] * dRR[8])

        A = tcon([RR, A], [[-1, -3, -5, -7, -9, 1], [-2, -4, -6, -8, 1]])
        D1 = size(A, 1)
        D2 = size(A, 2)
        A = reshape(A, D1 * D2, D1 * D2, D1 * D2, D1 * D2, size(A, 9))
    end

    A
end

function check_Q_op()
    Q_op = zeros(ComplexF64, 2, 2, 2, 2, 2)
    Q_op[1, 1, 1, :, :] = SI
    Q_op[1, 2, 2, :, :] = sigmax
    Q_op[2, 1, 2, :, :] = sigmay
    Q_op[2, 2, 1, :, :] = sigmaz

    # v = [0.0 1; im 0]
    v = [0.0 im; 1 0]

    # Q_op = abs.(Q_op)
    # v = abs.(v)

    @ein Qx[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, m4, p1] * sigmax[p1, m5]
    @ein Qy[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, m4, p1] * sigmay[p1, m5]
    @ein Qz[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, m4, p1] * sigmaz[p1, m5]

    @ein Qx2[m1, m2, m3, m4, m5] := v[p1, m2] * Q_op[m1, p1, p2, m4, m5] * (v')[m3, p2]
    @ein Qy2[m1, m2, m3, m4, m5] := v[m1, p1] * Q_op[p1, m2, p2, m4, m5] * (v')[p2, m3]
    @ein Qz2[m1, m2, m3, m4, m5] := v[p1, m1] * Q_op[p1, p2, m3, m4, m5] * (v')[m2, p2]

    display(v' * sigmaz * v)
    L = [Qx == Qx2, Qy == Qy2, Qz == Qz2]
    # display(Qx)
    # display(Qx2)

    L, Q_op
end

function check_R_op(phi)
    R_op = zeros(ComplexF64, 2, 2, 2, 2, 2)
    R_op[1, 1, 1, :, :] = SI .* cos(phi)
    R_op[2, 1, 1, :, :] = sigmax * sin(phi)
    R_op[1, 2, 1, :, :] = sigmay * sin(phi) 
    R_op[1, 1, 2, :, :] = sigmaz * sin(phi) 

    # v = [0.0 1; im 0]
    v = [0.0 im; 1 0]

    # Q_op = abs.(Q_op)
    # v = abs.(v)

    @ein Rx[m1, m2, m3, m4, m5] := R_op[m1, m2, m3, m4, p1] * sigmax[p1, m5]
    @ein Ry[m1, m2, m3, m4, m5] := R_op[m1, m2, m3, m4, p1] * sigmay[p1, m5]
    @ein Rz[m1, m2, m3, m4, m5] := R_op[m1, m2, m3, m4, p1] * sigmaz[p1, m5]

    @ein Rx2[m1, m2, m3, m4, m5] := v[p1, m2] * R_op[m1, p1, p2, m4, m5] * (v')[m3, p2]
    @ein Ry2[m1, m2, m3, m4, m5] := v[m1, p1] * R_op[p1, m2, p2, m4, m5] * (v')[p2, m3]
    @ein Rz2[m1, m2, m3, m4, m5] := v[p1, m1] * R_op[p1, p2, m3, m4, m5] * (v')[m2, p2]

    display(v' * sigmaz * v)
    L = [Rx == Rx2, Ry == Ry2, Rz == Rz2]
    # display(Rx)
    # display(Rx2)

    L, R_op
end