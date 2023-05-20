using iPEPS
using OMEinsum
using Zygote
using LinearAlgebra
A = [rand(5,5,5) for i = 1:4]
B = [rand(5,5,5) for i = 1:4];

function f(A,B)
    A1 = iPEPS.NestedTensor(A...)
    B1 = iPEPS.NestedTensor(B...);

    input = ((-1,1,2), (1,2,-2))
    out = (-1,-2)
    C = iPEPS.contract(A1, B1, input, out)
    C.T[1]
end
gradient(f, A, B)


function f1(A,B)
    A1 = iPEPS.NestedTensor(A...)
    B1 = iPEPS.NestedTensor(B...);
    IA = [-1, 1, 2]
    IB = [1,2, -2] 
    C = tensorcontract(A1, IA, B1, IB)
    C.T[1]
end
gradient(f1, A, B)


ts = [rand(5,5), rand(5,5), rand(5,5), rand(5,5)];
A = iPEPS.NestedTensor(ts);

B = deepcopy(A);
C = [A,B];
r0 =wrap_ncon([A,B], ((-1,1,2), (1,2,-2)), (-1,-2));
r1 =wrap_ncon(((-1,1,2), (1,2,-2)), (-1,-2) , A, B);


A, B, C = rand(5,5), rand(5,5), rand(5,5)
D = rand(5,5)
D = iPEPS.EmptyT()
ts = [A,B,C,D] |> iPEPS.NestedTensor

function f(ns, x)
    y = (ns*x)
    E = iPEPS.wrap_tr(y) #|> iPEPS.NestedTensor
    # E = iPEPS.tcon([ns, x], [[-1,1], [1,-2]])
    E[4] |> sum
end
f(ts, D)
gradient(x -> f(ts, x), D)


C1 = [rand(ComplexF64, 100,100) for i = 1:4] |> iPEPS.NestedTensor;
E4 = [rand(ComplexF64, 100,100,4,4) for i = 1:4] |> iPEPS.NestedTensor;

A = [rand(4, 4, 4, 4, 4), rand(4,4,4,4,4), iPEPS.EmptyT(), iPEPS.EmptyT()] |> iPEPS.NestedTensor;
Ad = [rand(4, 4, 4, 4, 4), iPEPS.EmptyT(), rand(4,4,4,4,4), iPEPS.EmptyT()] |> iPEPS.NestedTensor;


@time E4A = iPEPS.tcon([E4, A], [[-1, -3, 1, -6], [-2, 1, -4, -5, -7]]);
@time EAAd = iPEPS.tcon([E4A, Ad], [[-1, -2, -4, -5, -7, 1, 2], [-3, 1, -6, -8, 2]]);
    # @time EAAd = iPEPS.wrap_reshape(EAAd, prod(size(EAAd)[1:3]), prod(size(EAAd)[4:6]), size(EAAd,7), :)
    # @time EPd = iPEPS.tcon([EAAd, Pd], [[1, -2, -3, -4], [-1, 1]])
    # @time iPEPS.tcon([EPd, P], [[-1, 1, -3, -4], [1, -2]])

@time AA = iPEPS.tcon([A, Ad], [[-1, -3, -5, -7, 1], [-2, -4, -6, -8, 1]]);
@time EAA = iPEPS.tcon([E4, AA], [[-1, -4, 1, 2], [-2, -3, 1, 2, -5, -6, -7, -8]]);

@time C1E1 = iPEPS.tcon([C1, E1], [[-2, 1], [1, -1, -3, -4]]);
@time C1E1 = iPEPS.tcon([E1, C1], [[1, -1, -3, -4], [-2, 1]]);


@time EAAd = iPEPS.tcon([E4A, Ad], [[-1, -2, -4, -5, -7, 1, 2], [-3, 1, -6, -8, 2]]);

@time EAAd = iPEPS.tcon([Ad[1], E4A[1]], [[-3, 1, -6, -8, 2],[-1, -2, -4, -5, -7, 1, 2]]);

using TensorOperations
@time @tensor EAAd[:] := E4A[1][-1, -2, -4, -5, -7, 1, 2] * Ad[1][-3, 1, -6, -8, 2];