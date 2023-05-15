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