struct EmptyT end

Base.:(*)(A::EmptyT, B::EmptyT) = A
Base.:(*)(A::EmptyT, ::AbstractArray) = A
Base.:(*)(::AbstractArray, B::EmptyT) = B

renormalize(A::EmptyT) = A

