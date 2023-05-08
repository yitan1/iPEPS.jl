struct NestedTensor{AT} <: AbstractArray{AT, 1}
    data::Vector{AT}
end

Base.size(A::NestedTensor) = size(A.data)
Base.getindex(A::NestedTensor, i) = A.data[i]

# Base.length(A::NestedTensor) = prod(size(A))
# Base.IndexStyle(::NestedTensor) = IndexLinear()

# function TensorOperations.tensorcontract(A::NestedTensor, IA, B::NestedTensor, IB)
#     res1 = tensorcontract(A.T, IA, B.T, IB)
#     res2 = tensorcontract(A.T, IA, B.T, IB)
#     res3 = tensorcontract(A.T, IA, B.T, IB)
#     res4 = tensorcontract(A.T, IA, B.T, IB)

#     NestedTensor(res1, res2, res3, res4)
# end

renormalize(A::NestedTensor) = [A[i] ./ maximum(abs, A[i]) for i in eachindex(A)] |> NestedTensor

Base.:(*)(A::EmptyT, B::NestedTensor) = [A*B[i]  for i in eachindex(B)] |> NestedTensor
Base.:(*)(A::NestedTensor, B::EmptyT) = [A[i]*B  for i in eachindex(A)] |> NestedTensor

wrap_ncon(xs::Vector{NestedTensor{AT}}, ind_xs, ind_y) where AT = wrap_ncon(ind_xs, ind_y, xs...)

function wrap_ncon(ind_xs, ind_y, A::NestedTensor, B::NestedTensor)
    res1 = EinCode(ind_xs, ind_y)(A[1],B[1]) 
    res2 = EinCode(ind_xs, ind_y)(A[1],B[2]) .+ EinCode(ind_xs, ind_y)(A[2],B[1])
    res3 = EinCode(ind_xs, ind_y)(A[1],B[3]) .+ EinCode(ind_xs, ind_y)(A[3],B[1])
    res4 = EinCode(ind_xs, ind_y)(A[1],B[4]) .+ EinCode(ind_xs, ind_y)(A[4],B[1]) .+ 
                EinCode(ind_xs, ind_y)(A[2],B[3]) .+ EinCode(ind_xs, ind_y)(A[3],B[2])

    NestedTensor([res1, res2, res3, res4])
end

