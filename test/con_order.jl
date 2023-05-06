using iPEPS: tcon
X = 100
D = 4
d = 4

C4 = rand(X,X)
E4 = rand(X,X, D, D)
E3 = rand(X,X, D, D)
A = rand(D,D,D,D,d);

function f1(C4, E3, E4d, Ad, Add)
    C4E3 = tcon([C4,E3], [[-1,1], [1,-2,-3,-4]])
    CEE4 = tcon([C4E3, E4d], [[1,-2,-4,-6],[-1,1,-3,-5]])
    CEEA = tcon([CEE4, Ad], [[-1,-2,1,2,-3,-4], [-5,1,2,-6,-7]])
    tcon([CEEA, Add], [[-1,-4,1,2,-2,-5,3], [-3,1,2,-6,3]])
    nothing
end
function f2(C4, E3, E4d, Ad, Add)
    C4E3 = tcon([C4,E3], [[-1,1], [1,-2,-3,-4]])
    CEE4 = tcon([C4E3, E4d], [[1,-2,-6,-4],[-1,1,-5,-3]])
    CEEA = tcon([CEE4, Ad], [[-1,-2,-3,-4,1,2], [-5,1,2,-6,-7]])
    tcon([CEEA, Add], [[-1,-4,1,2,-2,-5,3], [-3,1,2,-6,3]])
    nothing
end
@time f1(C4, E3, E4, A, A)
@time f2(C4, E3, E4, A, A)

# function c1（）
using OMEinsum
allow_loops(false)
@time tcon([C4, E4, E3, A], [[1,2], [-1,1,3,-2], [2,-3, 4, -4], [3, 4, -5 ,-6, -7]]);

ein"(ij, aikb), jcmd, kmefg -> abcdefg"(C4, E4, E3, A);

function CEEA(C4, E4, E3, A)
    CE3 = tcon([C4, E3], [[-1,1], [1,-2, -3, -4]])
    CEE = tcon([E4, CE3], [[-1,1, -3,-4], [1, -2, -5, -6]])
    CEEA = tcon([CEE, A], [[-1,-2,1,-4,2,-5], [-3, 1, 2, -6,-7]])
    CEEA[1]
end

function CEEA(C4, E4, E3, A)
    CEEA = begin
        CE3 = tcon([C4, E3], [[-1,1], [1,-2, -3, -4]])
        CEE = tcon([E4, CE3], [[-1,1, -3,-4], [1, -2, -5, -6]])
        tcon([CEE, A], [[-1,-2,1,-4,2,-5], [-3, 1, 2, -6,-7]])
    end
    CEEA[1]
end

using Zygote
@time gradient(x -> CEEA(C4, E4, E3, x), A);
let
X = 100
D = 4
d = 4
A = rand(X^2*D^2, D^2)
B = rand(D^2, D^2*d)
C = rand(X*D*d, X*D^3)
D = rand(X*D^3, X*D*d);
begin
   @time A*B
   @time C*D
end
nothing
end

begin
@time CE3 = tcon([C4, E3], [[-1,1], [1,-2, -3, -4]]);
@time CEE = tcon([E4, CE3], [[-1,1, -3,-5], [1, -2, -4, -6]]);
@time tcon([CEE, A], [[-1,-2,1,2,-4,-5], [-3, 1, 2, -6,-7]]);
end;

begin
@time CE3 = tcon([C4, E3], [[-1,1], [1,-2, -3, -4]]);
@time CEE = tcon([E4, CE3], [[-1,1, -3,-4], [1, -2, -5, -6]]);
@time tcon([CEE, A], [[-1,-2,1,-4,2,-5], [-3, 1, 2, -6,-7]]);
end;

let
    X = 100
    D = 4
    d = 4 
A1 = rand(X^2, D^4)
B1 = rand(D^4, D^4*d^2)
C1 = rand(X*D^2*d^2, X*D^2)
D1 = rand(X*D^2, X);
begin
    @time A1*B1
    @time C1*D1
end
nothing
end
