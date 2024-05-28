function fprint(xs)
    println(xs)
    flush(stdout)
end
Base.print(io::IO, x::Float64) = @printf(io, "%.5g", x)
# function Base.print(io::IO, x::Float64) 
#     if abs(x) < 1e-4
#         return  @printf(io, "%0.6g", x)
#     else 
#         return @printf(io, "%0.4f", x)
#     end
# end
Base.print(io::IO, x::ComplexF64) = @printf(io, "%0.6f %0.6f im", real(x), imag(x))

function tcon(xs, ind_xs)
    ind_num = [count(<(0), ind_xs[i]) for i in eachindex(ind_xs)] |> sum
    ind_y = -collect(1:ind_num)
    tcon(ind_xs, ind_y, xs...)
end

# tcon(xs, ind_xs, ind_y) = EinCode(ind_xs, ind_y)(xs...)
# tcon(ind_xs, ind_y, xs...) = EinCode(ind_xs, ind_y)(xs...)
tcon(ind_xs, ind_y, A, B) = EinCode(ind_xs, ind_y)(A, B)

function wrap_svd(A, n=Inf)
    u, s, v = svd1(A)
    ix = sortperm(s)[end:-1:1]
    s = s[ix]
    u = u[:, ix]
    v = v[:, ix]

    u, s, v = cutoff_matrix(u, s, v, 1e-12, n)

    u, s, v
end

function cutoff_matrix(u, s, v, cutoff, n)
    n_above_cutoff = count(>(cutoff), s / maximum(s))
    n = min(n, n_above_cutoff) |> Int

    if n < Inf && size(s, 1) > n
        # if  s[n] > 1e-6
        #     new_n = count(>=(s[n] - cutoff), s)
        #     n = min(new_n, n + 10)
        # end
        if abs(s[n+1] - s[n]) < 1e-10
            n = n + 1
        end
        u = u[:, 1:n]
        s = s[1:n]
        v = v[:, 1:n]
    end

    u, s, v
end

renormalize(A::AbstractArray) = A ./ maximum(abs.(A))
# renormalize(A::AbstractArray) = A ./ norm(A)


function diag_inv(A::AbstractArray)
    diagm(1 ./ A)
end