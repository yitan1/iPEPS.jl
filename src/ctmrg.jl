
# function rg_step(tensors::CTMTensors)
#     tensors, s = left_rg(tensors)
#     tensors, _ = right_rg(tensors)
#     tensors, _ = top_rg(tensors)
#     tensors, _ = bottom_rg(tensors)

#     tensors, s
# end

# function left_rg(ts)
#     A = ts.A
#     a
# end

using TensorOperations
function init_CE(A)
    @tensor T[m1,m2,m3,m4,m5,m6,m7,m8] = A[p1, m1, m3, m5, m7]*conj(A)[p1, m2, m4, m6, m8]
    @tensor C1[m1, m2] := T[p1, p1, ]
end