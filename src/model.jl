const Sx = Float64[0 1; 1 0]/2
const Sy = ComplexF64[0 -1im; 1im 0]/2
const Sz = Float64[1 0; 0 -1]/2
const SI = Float64[1 0; 0 1]
    # sp = [0 1; 0 0]
    # sm = [0 0; 1 0]

function honeycomb(Jx = 1, Jy = 1)
    hh = Jx*2*kron(kron(SI, Sx), kron(Sx, SI)) .+ ( kron(kron(Sz, Sz), kron(SI, SI) ) .+ kron(kron(SI, SI), kron(Sz, Sz)) ) / 2  .|> real
    hv = Jy*2*kron(kron(SI, Sy), kron(Sy, SI)) .+ ( kron(kron(Sz, Sz), kron(SI, SI) )  .+ kron(kron(SI, SI), kron(Sz, Sz)) ) / 2  .|> real

    [-hh, -hv]
end

function init_gs()
    Q_op = zeros(ComplexF64,2,2,2,2,2)
    Q_op[1,1,1,:,:] = SI
    Q_op[1,2,2,:,:] = Sx
    Q_op[2,1,2,:,:] = Sy
    Q_op[2,2,1,:,:] = Sz
    ux, uy, uz = 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)
    s111 = 1/sqrt(2+2*uz)*[1 + uz, ux + im*uy]

    T = tcon([Q_op, s111], [[-1,-2,-3,-4,1], [1]])
    # r = tcon([Q_op, s111], [[-3,-2,-1,-4,1], [1]])
    # A = tcon([T, T], [[1,-1,-2,-5], [1,-3,-4,-6]]) # XX
    # A = tcon([T, T], [[-1,1,-4,-5], [-3,1,-2,-6]]) # YY
    A = tcon([T, T], [[-1,-2,1,-5], [-3,-4,1,-6]]) # ZZ
    A = reshape(A, 2, 2, 2, 2, 4)
    A
end