function convert_order_to(ts)
    C1, C2, C3, C4 = ts.Cs
    T1, T2, T3, T4 = ts.Es
    A = ts.A
    D = size(A,1)
    
    C10 = permutedims(C1, [2,1])
    C20 = permutedims(C4, [2,1])
    C30 = permutedims(C3, [2,1])
    C40 = C2

    T10 = reshape(T4, size(T4, 1), size(T4, 2), D*D)

    T20 = reshape(T3, size(T3, 1), size(T3, 2), D*D)

    T30 = reshape(T2, size(T2, 1), size(T2, 2), D*D)
    T30 = permutedims(T30, [2,1,3])

    T40 = reshape(T1, size(T1, 1), size(T1, 2), D*D)
    T40 = permutedims(T40, [2,1,3])

    ts = setproperties(ts, Cs=[C10, C20, C30, C40], Es=[T10, T20, T30, T40])

    ts
end
function convert_order_back(ts)
    C1, C2, C3, C4 = ts.Cs
    T1, T2, T3, T4 = ts.Es
    A = ts.A

    D = size(A,1)

    C10 = permutedims(C1, [2,1])
    C30 = permutedims(C3, [2,1])
    C20 = permutedims(C4, [2,1])
    C40 = C2
    
    T10 = begin
        T10 = permutedims(T4, [2,1,3])
        reshape(T10, size(T10, 1), size(T10, 2), D, D)
    end

    T20 = begin
        T20 = permutedims(T3, [2,1,3])
        reshape(T20, size(T20, 1), size(T20, 2), D, D)
    end

    T30 = reshape(T2, size(T2, 1), size(T2, 2), D, D)

    T40 = reshape(T1, size(T1, 1), size(T1, 2), D, D)

    ts = setproperties(ts, Cs=[C10, C20, C30, C40], Es=[T10, T20, T30, T40])

    ts
end

