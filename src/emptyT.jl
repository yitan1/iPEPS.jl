struct EmptyT end

Base.:(*)(A::EmptyT, B) = A
Base.:(*)(A::EmptyT, ::EmptyT) = A
Base.:(*)(A::EmptyT, ::AbstractArray) = A
Base.:(*)(::AbstractArray, B::EmptyT) = B

Base.:(+)(A::EmptyT, ::EmptyT) = A
Base.:(+)(::EmptyT, B::AbstractArray) = B
Base.:(+)(A::AbstractArray, ::EmptyT) = A
Base.:(-)(A::EmptyT) = A

Base.reshape(A::EmptyT, ::Vararg{Union{Colon, Int64}}) = A
Base.reshape(A::EmptyT, ::Tuple) = A
Base.size(::EmptyT) = (0,)
Base.size(A::EmptyT, ::Int64) = 0
Base.permutedims(A::EmptyT, perm) = A
Base.length(A::EmptyT) = 0
Base.conj(A::EmptyT) = A
Base.abs(A::EmptyT) = 0.0
Base.maximum(f, A::EmptyT) = 0.0
Base.sum(A::EmptyT) = 0.0

tcon(ind_xs, ind_y, A::EmptyT, ::EmptyT) = A
tcon(ind_xs, ind_y, A::EmptyT, B) = A
tcon(ind_xs, ind_y, A, B::EmptyT) = B

renormalize(A::EmptyT) = A

