export test

function test() #compute wilson loop in honeycomb lattice
    
end

function get_vison(D; p1=0.24, p2=0.0)
    Q_op = zeros(ComplexF64,2,2,2,2,2)
    Q_op[1,1,1,:,:] = SI
    Q_op[1,2,2,:,:] = sigmax
    Q_op[2,1,2,:,:] = sigmay
    Q_op[2,2,1,:,:] = sigmaz

    ux, uy, uz = 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)
    s111 = 1/sqrt(2+2*uz)*[1 + uz, ux + im*uy]

    T = tcon([Q_op, s111], [[-1,-2,-3,-4,1], [1]])

    @ein T1[m1,m2,m3,m4] := T[m1,m2,p1,m4]*sigmaz[p1,m3]
    A = tcon([T1, T], [[-1,-2,1,-5], [-3,-4,1,-6]]) # ZZ
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

function act_Q_op(A0)
    Q_op = zeros(ComplexF64,2,2,2,2,2)
    Q_op[1,1,1,:,:] = SI
    Q_op[1,2,2,:,:] = sigmax
    Q_op[2,1,2,:,:] = sigmay
    Q_op[2,2,1,:,:] = sigmaz

    QQ = tcon([Q_op, Q_op], [[-1,-2,1,-5,-7], [-3,-4,1,-6,-8]])
    dim = size(QQ)
    QQ = reshape(QQ, dim[1],dim[2], dim[3],dim[4], dim[5]*dim[6], dim[7]*dim[8])

    A = tcon([QQ,A0], [[-1, -3, -5, -7, -9, 1], [-2, -4, -6, -8, 1]])
    D1 = size(A,1)
    D2 = size(A,2)
    A = reshape(A, D1*D2, D1*D2, D1*D2, D1*D2, size(A,9))

    A
end

