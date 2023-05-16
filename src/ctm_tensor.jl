TensorType = Union{AbstractArray, EmptyT}

struct CTMTensors{AT, CT, ET, BT, B_CT, B_ET} 
    A::AT
    Ad::AT
    Cs::Vector{CT}
    Es::Vector{ET}
    B::BT
    Bd::BT
    B_Cs::Vector{B_CT}
    Bd_Cs::Vector{B_CT}
    BB_Cs::Vector{B_CT}
    B_Es::Vector{B_ET}
    Bd_Es::Vector{B_ET}
    BB_Es::Vector{B_ET}

    Params::Dict{String, Any}
end

function get_all_A(ts::CTMTensors)
    if ts.B isa EmptyT
        return ts.A
    else 
        nested_A = [ts.A, ts.B, EmptyT(), EmptyT()] |> NestedTensor
        nested_A
    end
end

function get_all_Ad(ts::CTMTensors)
    if ts.B isa EmptyT
        return ts.Ad
    else 
        nested_Ad = [ts.Ad, EmptyT(), ts.Bd, EmptyT()] |> NestedTensor
        nested_Ad
    end
end

function get_all_Cs(ts::CTMTensors)
    if ts.B isa EmptyT
        return ts.Cs
    else 
        nested_Cs = [ [ts.Cs[i], ts.B_Cs[i], ts.Bd_Cs[i], ts.BB_Cs[i]] |> NestedTensor for i=1:4]
        return nested_Cs
    end
end

function get_all_Es(ts::CTMTensors)
    if ts.B isa EmptyT
        return ts.Es
    else 
        nested_Es = [ [ts.Es[i], ts.B_Es[i], ts.Bd_Es[i], ts.BB_Es[i]] |> NestedTensor for i = 1:4]
        return nested_Es
    end

end

CTMTensors(A) = CTMTensors(A, conj(A))
function CTMTensors(A, Ad)
    Cs, Es = init_ctm(A, Ad)
    B = EmptyT()
    B_C = [B for i = 1:4]
    Params = TOML.parsefile("$(@__DIR__)/config.toml")
    CTMTensors(A, Ad, Cs, Es, B, B, B_C, B_C, B_C, B_C, B_C, B_C, Params)
end

function init_ctm(A, Ad)
    D = size(A,1)
    C1 = ones(1,1)
    C2 = ones(1,1)
    C3 = ones(1,1)
    C4 = ones(1,1)
    Cs = Vector{typeof(C1)}([C1, C2, C3, C4]) #.|> renormalize

    E1 = tcon([A,Ad], [[1,2,-1,3,4],[1,2,-2,3,4]])
    E1 = reshape(E1, 1, 1, D, D)

    E2 = tcon([A,Ad], [[1,-1,2,3,4],[1,-2,2,3,4]])
    E2 = reshape(E2, 1, 1, D, D)

    E3 = tcon([A,Ad], [[-1,1,2,3,4],[-2,1,2,3,4]])
    E3 = reshape(E3, 1, 1, D, D)

    E4 = tcon([A,Ad], [[1,2,3,-1,4],[1,2,3,-2,4]])
    E4 = reshape(E4, 1, 1, D, D)

    Es = Vector{typeof(E1)}([E1, E2, E3, E4]) #.|> renormalize

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