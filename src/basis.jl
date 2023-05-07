renormlize(A::AbstractArray) = A ./ maximum(abs, A)

