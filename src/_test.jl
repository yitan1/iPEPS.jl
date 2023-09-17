export test

function test()
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

function check_Q_op()
    Q_op = zeros(ComplexF64,2,2,2,2,2)
    Q_op[1,1,1,:,:] = SI
    Q_op[1,2,2,:,:] = sigmax
    Q_op[2,1,2,:,:] = sigmay
    Q_op[2,2,1,:,:] = sigmaz

    # v = [0.0 1; im 0]
    v = [0.0 im; 1 0]

    # Q_op = abs.(Q_op)
    # v = abs.(v)

    @ein Qx[m1,m2,m3,m4,m5] := Q_op[m1,m2,m3,m4,p1]*sigmax[p1,m5]
    @ein Qy[m1,m2,m3,m4,m5] := Q_op[m1,m2,m3,m4,p1]*sigmay[p1,m5]
    @ein Qz[m1,m2,m3,m4,m5] := Q_op[m1,m2,m3,m4,p1]*sigmaz[p1,m5]

    @ein Qx2[m1,m2,m3,m4,m5] := v[p1,m2]*Q_op[m1,p1,p2,m4,m5]*(v')[m3, p2]
    @ein Qy2[m1,m2,m3,m4,m5] := v[m1,p1]*Q_op[p1,m2,p2,m4,m5]*(v')[p2, m3]
    @ein Qz2[m1,m2,m3,m4,m5] := v[p1,m1]*Q_op[p1,p2,m3,m4,m5]*(v')[m2, p2]

    display(v' *sigmaz*v)
    L = [Qx ==Qx2, Qy ==Qy2, Qz ==Qz2]    
    # display(Qx)
    # display(Qx2)

    L
end