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