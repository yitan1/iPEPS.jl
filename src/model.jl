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
    H = -tout(sigmax, sigmax) .- h * tout(sigmaz, SI) / 2 / 2 .- h * tout(SI, sigmaz) / 2 / 2
    # H = -tout(Sz, Sz) .+ h * tout(Sx, SI) / 2 / 2 .+ h * tout(SI, Sx) / 2 / 2

    [H, H]
end

function heisenberg(Jz=1)
    H = Jz * tout(Sz, Sz) - tout(Sx, Sx) - tout(Sy, Sy)

    [H, H]
end

function honeycomb(Jx=1, Jy=1; Jz = 1, dir = "ZZ")
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

# TODO
function init_hb_gs(D=4; p1=0.24, p2=0.0, dir = "ZZ")
    Q_op = zeros(ComplexF64, 2, 2, 2, 2, 2)
    Q_op[1, 1, 1, :, :] = SI
    Q_op[1, 2, 2, :, :] = Sx * 2
    Q_op[2, 1, 2, :, :] = Sy * 2
    Q_op[2, 2, 1, :, :] = Sz * 2
    ux, uy, uz = 1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)
    s111 = 1 / sqrt(2 + 2 * uz) * [1 + uz, ux + im * uy]

    T = tcon([Q_op, s111], [[-1, -2, -3, -4, 1], [1]])
    if dir == "ZZ"
        A = tcon([T, T], [[-1, -2, 1, -5], [-3, -4, 1, -6]]) # ZZ
    elseif dir == "XX"
        A = tcon([T, T], [[1,-1,-2,-5], [1,-3,-4,-6]]) # XX
    elseif dir == "YY"
        A = tcon([T, T], [[-2, 1, -1, -5], [-4, 1, -3, -6]]) # YY
    end

    A = reshape(A, 2, 2, 2, 2, 4)

    if D == 4
        A = act_R_op(A, p1 = p1, dir = dir)
    end

    if D == 8
        A = act_R_op(A, p1 = p1, dir = dir)
        A = act_R_op(A, p1 = p2, dir = dir)
    end

    A
end

function act_R_op(A0; p1 = 0.24, dir = "ZZ")
    phi = p1 * pi
    R_op = zeros(ComplexF64, 2, 2, 2, 2, 2)
    R_op[1, 1, 1, :, :] = SI .* cos(phi)
    R_op[2, 1, 1, :, :] = 2 * Sx * sin(phi) 
    R_op[1, 2, 1, :, :] = 2 * Sy * sin(phi) 
    R_op[1, 1, 2, :, :] = 2 * Sz * sin(phi) 

    if dir == "ZZ"
        RR = tcon([R_op, R_op], [[-1, -2, 1, -5, -7], [-3, -4, 1, -6, -8]]) # ZZ
    elseif dir == "XX"
        RR = tcon([R_op, R_op], [[1,-1,-2,-5,-7], [1,-3,-4,-6,-8]]) # XX
    elseif dir == "YY"
        RR = tcon([R_op, R_op], [[-2, 1, -1, -5, -7], [-4, 1, -3, -6, -8]]) # YY
    end
    dRR = size(RR)
    RR = reshape(RR, dRR[1], dRR[2], dRR[3], dRR[4], dRR[5] * dRR[6], dRR[7] * dRR[8])

    A = tcon([RR, A0], [[-1, -3, -5, -7, -9, 1], [-2, -4, -6, -8, 1]])
    D1 = size(A, 1)
    D2 = size(A, 2)
    A = reshape(A, D1 * D2, D1 * D2, D1 * D2, D1 * D2, size(A, 9))
end

function get_Q_op()
    Q_op = zeros(ComplexF64, 2, 2, 2, 2, 2)
    Q_op[1, 1, 1, :, :] = SI
    Q_op[1, 2, 2, :, :] = sigmax
    Q_op[2, 1, 2, :, :] = sigmay
    Q_op[2, 2, 1, :, :] = sigmaz

    Q_op
end

function get_ghz()
    ghz = zeros(ComplexF64, 2, 2, 2, 2)
    ghz[1, 1, 1, 1] = 1
    ghz[2, 2, 2, 2] = 1

    ghz
end

function get_ghz_111()
    ghz = get_ghz()

    phi = pi / 4
    cost2 = sqrt((1+1/sqrt(3))/2)
    sint2 = sqrt((1-1/sqrt(3))/2)
    S = [exp(-im * phi/2) * cost2 -exp(-im * phi/2) * sint2; exp(im * phi/2) * sint2 exp(im * phi/2) * cost2] 

    @ein ghz[m1, m2, m3, m4] := ghz[m1, m2, m3, p1] * S[m4, p1]

    ghz, S
end

function get_Q_ghz()
    Q_op = get_Q_op()

    QQ = tcon([Q_op, Q_op], [[-1, -2, 1, -5, -7], [-3, -4, 1, -6, -8]])
    dQQ = size(QQ)
    QQ = reshape(QQ, dQQ[1], dQQ[2], dQQ[3], dQQ[4], dQQ[5] * dQQ[6], dQQ[7] * dQQ[8])

    ghz, _ = get_ghz_111()
    gg = tcon([ghz, ghz], [[-1, -2, 1, -5], [-3, -4, 1, -6]])
    gg = reshape(gg, 2, 2, 2, 2, 4)

    A = tcon([QQ, gg], [[-1, -3, -5, -7, -9, 1], [-2, -4, -6, -8, 1]])
    D1 = size(A, 1)
    D2 = size(A, 2)
    A = reshape(A, D1 * D2, D1 * D2, D1 * D2, D1 * D2, size(A, 9))

    A
end

function check_Q_op()
    Q_op = get_Q_op()

    v = [0.0 im; 1 0]
    vt = [0.0 1; im 0]
    vvt = sigmaz

    # Q * sigma
    # @ein Qx[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, m4, p1] * sigmax[p1, m5] 
    # @ein Qy[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, m4, p1] * sigmay[p1, m5] 
    # @ein Qz[m1, m2, m3, m4, m5] := Q_op[m1, m2, m3, m4, p1] * sigmaz[p1, m5]

    # @ein Qx2[m1, m2, m3, m4, m5] := vt[m2, p1] * Q_op[m1, p1, p2, m4, m5] * (vt')[p2, m3]
    # @ein Qy2[m1, m2, m3, m4, m5] := vt[m3, p2] * Q_op[p1, m2, p2, m4, m5] * (vt')[p1, m1]
    # @ein Qz2[m1, m2, m3, m4, m5] := vt[m1, p1] * Q_op[p1, p2, m3, m4, m5] * (vt')[p2, m2]

    # sigma * Q * sigma
    @ein Qx[m1, m2, m3, m4, m5] := sigmax[m4, p1] * Q_op[m1, m2, m3, p1, p2] * sigmax[p2, m5]
    @ein Qy[m1, m2, m3, m4, m5] := sigmay[m4, p1] * Q_op[m1, m2, m3, p1, p2] * sigmay[p2, m5]
    @ein Qz[m1, m2, m3, m4, m5] := sigmaz[m4, p1] * Q_op[m1, m2, m3, p1, p2] * sigmaz[p2, m5]

    @ein Qx2[m1, m2, m3, m4, m5] := vvt[m2, p1] * Q_op[m1, p1, p2, m4, m5] * (vvt')[p2, m3]
    @ein Qy2[m1, m2, m3, m4, m5] := vvt[m3, p2] * Q_op[p1, m2, p2, m4, m5] * (vvt')[p1, m1]
    @ein Qz2[m1, m2, m3, m4, m5] := vvt[m1, p1] * Q_op[p1, p2, m3, m4, m5] * (vvt')[p2, m2]

    # display(v' * sigmaz * v)
    L = [Qx == Qx2, Qy == Qy2, Qz == Qz2]
    # display(Qz)
    # display(Qz2)

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