struct EmptyT end

Base.:(*)(A::EmptyT, B) = A
Base.:(*)(A::EmptyT, ::EmptyT) = A
Base.:(*)(A::EmptyT, ::AbstractArray) = A
Base.:(*)(::AbstractArray, B::EmptyT) = B

Base.:(+)(A::EmptyT, ::EmptyT) = A
Base.:(+)(::EmptyT, B::AbstractArray) = B
Base.:(+)(A::AbstractArray, ::EmptyT) = A

Base.reshape(A::EmptyT, ::Vararg{Union{Colon, Int64}}) = A
Base.size(::EmptyT) = (0,)
Base.size(A::EmptyT, ::Int64) = 0
Base.permutedims(A::EmptyT, perm) = A

tcon(ind_xs, ind_y, A::EmptyT, ::EmptyT) = A
tcon(ind_xs, ind_y, A::EmptyT, B) = A
tcon(ind_xs, ind_y, A, B::EmptyT) = B

renormalize(A::EmptyT) = A

