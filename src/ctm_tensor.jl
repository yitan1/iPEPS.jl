struct CTMTensors
    A
    Ad
    Cs
    Es
    B
    Bd
    B_Cs
    Bd_Cs
    BB_Cs
    B_Es
    Bd_Es
    BB_Es
end

function CTMTensors(A, Ad)
    Cs, Es = init_ctm(A,Ad)
    t0 = EmptyT()
    CTMTensors(A, Ad, Cs, Es, t0, t0, t0, t0, t0, t0, t0, t0)
end

function init_ctm(A, Ad)
    D = size(A,1)
    C1 = ones(1,1)
    C2 = ones(1,1)
    C3 = ones(1,1)
    C4 = ones(1,1)
    Cs = [C1, C2, C3, C4]

    E1 = tcon([A,Ad], [[1,2,-1,3,4],[1,2,-2,3,4]])
    E1 = reshape(E1, 1, 1, D, D)

    E2 = tcon([A,Ad], [[1,-1,2,3,4],[1,-2,2,3,4]])
    E2 = reshape(E2, 1, 1, D, D)

    E3 = tcon([A,Ad], [[-1,1,2,3,4],[-2,1,2,3,4]])
    E3 = reshape(E3, 1, 1, D, D)

    E4 = tcon([A,Ad], [[1,2,3,-1,4],[1,2,3,-2,4]])
    E4 = reshape(E4, 1, 1, D, D)

    Es = [E1, E2, E3, E4]

    Cs, Es
end

function init_ctm1(A, Ad)
    D = size(A,1)
    C1 = tcon([A,Ad], [[1,2,-1,-3,3],[1,2,-2,-4,3]])
    C1 = reshape(C1, D*D, D*D)

    C2 = tcon([A,Ad], [[1,-1,-3,2,3],[1,-2,-4,2,3]])
    C2 = reshape(C2, D*D, D*D)

    C3 = tcon([A,Ad], [[-1,-3,1,2,3],[-2,-4,1,2,3]])
    C3 = reshape(C3, D*D, D*D)

    C4 = tcon([A,Ad], [[-1,1,2,-3,3],[-2,1,2,-4,3]])
    C4 = reshape(C4, D*D, D*D)

    Cs = (C1, C2, C3, C4) ./ norm.((C1, C2, C3, C4))

    E1 = tcon([A,Ad], [[1,-1,-5,-3,2],[1,-2,-6,-4,2]])
    E1 = reshape(E1, D*D, D*D, D, D)

    E2 = tcon([A,Ad], [[-1,-5,-3,1,2],[-2,-6,-4,1,2]])
    E2 = reshape(E2, D*D, D*D, D, D)

    E3 = tcon([A,Ad], [[-5,-1,1,-3,2],[-6,-2,1,-4,2]])
    E3 = reshape(E3, D*D, D*D, D, D)

    E4 = tcon([A,Ad], [[-1,1,-3,-5,2],[-2,1,-4,-6,2]])
    E4 = reshape(E4, D*D, D*D, D, D)

    Es = (E1, E2, E3, E4) ./ norm.((E1, E2, E3, E4))

    Cs, Es
end