# keep symmetry
get_symmetry(T) = get_symmetry(T, get_gz(size(T, 1)))
function get_symmetry(T, gz)
    q0 = randn(size(gz))
    Q, _ = qr(q0)
    q = Matrix(Q)

    maxiter = 100
    for i = 1:maxiter
        result = optimize(x -> obj_fun(x, T, gz), q, BFGS())
        q1 = result.minimizer
        @ein new_T[m1, m2, m3, m4, m5] := T[p1, p2, p3, p4, m5] * q1[m1, p1] * (q1')[m2, p2] * (q1')[m3, p3] * q1[m4, p4]

        delta = result.minimum / norm(new_T)^2

        if delta < 1e-5
            new_T[findall(x -> x < eps(), new_T)] .= 0.0
            return new_T
        else
            i == maxiter
            error("can not find symmetrical T")
        end
    end
end

function obj_fun(U, T, gz)
    new_gz1 = gz * U
    new_gz2 = gz * U'
    @ein T1[m1, m2, m3, m4, m5] := T[p1, p2, p3, p4, m5] * U[m1, p1] * (U')[m2, p2] * (U')[m3, p3] * U[m4, p4]
    @ein T2[m1, m2, m3, m4, m5] := T[p1, p2, p3, p4, m5] * new_gz1[m1, p1] * new_gz2[m2, p2] * new_gz2[m3, p3] * new_gz1[m4, p4]

    delta = sum(abs2, T1 .- T2)

    delta
end

function get_gz(D)
    gz = zeros(D, D)
    for i in axes(gz, 1)
        if i < div(D, 2) + 1
            gz[i, i] = 1
        else
            gz[i, i] = -1
        end
    end

    gz
end