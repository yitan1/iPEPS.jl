using iPEPS
using Zygote
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


ts = [rand(5,5,5), rand(5,5,5), rand(5,5,5), rand(5,5,5)];
A = NestedTensor(ts);
B = deepcopy(A);
C = [A,B];
r0 =wrap_ncon([A,B], ((-1,1,2), (1,2,-2)), (-1,-2));
r1 =wrap_ncon(((-1,1,2), (1,2,-2)), (-1,-2) , A, B);


A, B, C = rand(5,5,5,5), rand(5,5,5), rand(5,5,5);
wrapped_ncon([A,B,C], ( (-1,-2,1,2), (1,2,3) , (3,-3,-4)), (-1,-2,-3,-4));
