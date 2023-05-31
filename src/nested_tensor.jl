struct NestedTensor{AT}
    data::Vector{AT}
end

Base.eachindex(A::NestedTensor) = eachindex(A.data)
Base.size(A::NestedTensor) = size(A.data[1])
Base.size(A::NestedTensor, i) = size(A.data[1], i)
Base.getindex(A::NestedTensor, i) = A.data[i]

Base.real(A::NestedTensor) = real.(A.data) |> NestedTensor

Base.:(*)(a, B::NestedTensor) = [a*B[i]  for i in eachindex(B)] |> NestedTensor
Base.:(*)(A::NestedTensor, b) = [A[i]*b  for i in eachindex(A)] |> NestedTensor
# Base.:(*)(a::AbstractMatrix, B::NestedTensor) = [a*B[i]  for i in eachindex(B)] |> NestedTensor
# Base.:(*)(A::NestedTensor, b::AbstractMatrix) = [A[i]*b  for i in eachindex(A)] |> NestedTensor
# Base.:(*)(A::NestedTensor, B::NestedTensor) = [A[i]*B[i]  for i in eachindex(A)] |> NestedTensor

Base.:(/)(A::NestedTensor, B::NestedTensor) = [A[i] / B[i]  for i in eachindex(A)] |> NestedTensor

Base.:(+)(A::NestedTensor, B::NestedTensor) = [A[i] .+ B[i]  for i in eachindex(A)] |> NestedTensor
Base.:(-)(A::NestedTensor) = [ -A[i] for i in eachindex(A) ] |> NestedTensor 

wrap_reshape(A::AbstractArray, dims::Vararg{Union{Colon, Int64}}) = reshape(A, dims)
wrap_reshape(A::NestedTensor, dims::Vararg{Union{Colon, Int64}}) = [reshape(A[i], dims...) for i = eachindex(A)] |> NestedTensor

wrap_permutedims(A::AbstractArray, perm) = permutedims(A, perm)
wrap_permutedims(A::NestedTensor, perm) = [permutedims(A[i], perm) for i = eachindex(A)] |> NestedTensor

wrap_tr(A) = tr(A)
wrap_tr(A::NestedTensor) = [tr(A[i]) for i = eachindex(A)] |> NestedTensor
# LinearAlgebra.tr(A::NestedTensor) = tr.(A.data) |> NestedTensor
# @Zygote.adjoint function LinearAlgebra.tr(A::NestedTensor)
#     d = size(A, 1)
#     back(Δ) = (NestedTensor(Diagonal(fill(Δ, d)) for i = 1:4) ,)
#     return tr(A), back
# end

# renormalize(A::NestedTensor) = [renormalize(A[i]) for i in eachindex(A)] |> NestedTensor
function renormalize(A::NestedTensor) 
    factor = maximum(abs.(A[1]))
    A = A * (1 / factor)

    A        
end

function tcon(ind_xs, ind_y, A::NestedTensor, B::NestedTensor)
    res1 = tcon(ind_xs, ind_y, A[1], B[1])

    res2 = tcon(ind_xs, ind_y, A[1], B[2]) + tcon(ind_xs, ind_y, A[2], B[1])

    res3 = tcon(ind_xs, ind_y, A[1], B[3]) + tcon(ind_xs, ind_y, A[3], B[1])

    res4 = tcon(ind_xs, ind_y, A[1], B[4]) + tcon(ind_xs, ind_y, A[4], B[1]) + 
           tcon(ind_xs, ind_y, A[2], B[3]) + tcon(ind_xs, ind_y, A[3], B[2])

    NestedTensor([res1, res2, res3, res4])
end

tcon(ind_xs, ind_y, A::NestedTensor, B::AbstractArray) = [tcon(ind_xs, ind_y, A[i], B) for i in eachindex(A)] |> NestedTensor

tcon(ind_xs, ind_y, A::AbstractArray, B::NestedTensor) = [tcon(ind_xs, ind_y, A, B[i]) for i in eachindex(B)] |> NestedTensor

shift(A::NestedTensor, phi) = [A[1], A[2]*exp(im*phi), A[3]*exp(-im*phi), A[4]] |> NestedTensor

# shift(A, phi) = A