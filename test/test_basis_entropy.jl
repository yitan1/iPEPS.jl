using Test
using iPEPS
using iPEPS
using OMEinsum
using TOML
using ConstructionBase

function get_dm2(C1, C2, C3, C4, E1, E2, E3, E4)
    C1E1 = tcon([C1, E1], [[-1,1], [1,-2,-3,-4]])
    CEE1 = tcon([C1E1, E1], [[-1,1,-3,-5], [1,-2,-4,-6]])
    CEEC2 = tcon([CEE1, C2], [[-1,1,-3,-4,-5,-6], [1,-2]])
    CEECE4 = tcon([CEEC2, E4], [[1,-2,-3,-4,-6,-7], [1,-1,-5,-8]])

    C4E3 = tcon([C4, E3], [[-1, 1], [1, -2, -3, -4]])
    CEE3 = tcon([C4E3, E3], [[-1, 1, -3, -5], [1, -2, -4, -6]])
    CEEC3 = tcon([CEE3, C3], [[-1,1,-3,-4, -5, -6], [-2,1]])
    CEECE2 = tcon([CEEC3, E2], [[-1,1,-3,-4,-6,-7], [-2,1,-5,-8]])

    n_dm2 = tcon([CEECE4, CEECE2], [[1,2, -1,-2,-3, -7,-8,-9,], [1,2, -4,-5,-6, -10, -11, -12]])

    n_dm2
end

function get_dm3(C1, C2, C3, C4, E1, E2, E3, E4)
    C1E1 = tcon([C1, E1], [[-1,1], [1,-2,-3,-4]])
    CEE1 = tcon([C1E1, E1], [[-1,1,-3,-5], [1,-2,-4,-6]])
    CEEE1 = tcon([CEE1, E1], [[-1,1, -3,-4, -6,-7], [1,-2, -5,-8]])
    CEEEC2 = tcon([CEEE1, C2], [[-1,1, -3,-4,-5, -6,-7,-8], [1,-2]])
    CEEECE4 = tcon([CEEEC2, E4], [[1,-2, -3,-4,-5, -7,-8,-9], [1,-1,-6,-10]])

    C4E3 = tcon([C4, E3], [[-1, 1], [1, -2, -3, -4]])
    CEE3 = tcon([C4E3, E3], [[-1, 1, -3, -5], [1, -2, -4, -6]])
    CEEE3 = tcon([CEE3, E3], [[-1,1, -3,-4, -6,-7], [1,-2, -5,-8]])
    CEEEC3 = tcon([CEEE3, C3], [[-1,1, -3,-4,-5, -6,-7,-8], [-2,1]])
    CEEECE2 = tcon([CEEEC3, E2], [[-1,1, -3,-4,-5, -7,-8,-9], [-2,1,-6,-10]])

    n_dm3 = tcon([CEEECE4, CEEECE2], [[1,2, -1,-2,-3,-4, -9,-10,-11,-12], [1,2, -5,-6,-7,-8, -13,-14,-15,-16]])

    n_dm3
end

# function get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)
#     C1E1 = tcon([C1, E1], [[-1,1], [1,-2,-3,-4]])
#     CEE1 = tcon([C1E1, E1], [[-1,1,-3,-5], [1,-2,-4,-6]])

#     C2E2 = tcon([C2, E2], [[-1,1], [1,-2,-3,-4]])
#     CEE2 = tcon([C2E2, E2], [[-1,1,-3,-5], [1,-2,-4,-6]])

#     C3E3 = tcon([C3, E3], [[-1,1], [-2,1,-3,-4]])
#     CEE3 = tcon([C3E3, E3], [[-1,1,-4,-6], [-2,1,-3,-5]])

#     C4E4 = tcon([C4, E4], [[1,-2], [-1,1,-3,-4]])
#     CEE4 = tcon([C4E4, E4], [[1,-2,-4,-6], [-1,1,-3,-5]])

#     n_dm4u = tcon([CEE1, CEE4], [[1,-1, -3,-4, -7, -8], [1,-2, -5,-6, -9, -10]])
#     n_dm4d = tcon([CEE3, CEE2], [[1,-2, -3,-4, -7, -8], [-1,1, -5,-6, -9, -10]])

#     n_dm4 = tcon([n_dm4u, n_dm4d], [[1,2, -1,-2,-3,-4, -9,-10,-11,-12], [1,2, -5,-6,-7,-8, -13,-14,-15,-16]])

#     n_dm4
# end


function diag_n_dm(ts::CTMTensors)
    # A = ts.A
    # Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_single_dm(C1, C2, C3, C4, E1, E2, E3, E4)

    n_dm = reshape(n_dm, prod(size(n_dm)[1:4]), :)
    n_dm = (n_dm + n_dm')/2

    vs, vecs = eigen(n_dm)

    vs, vecs
end

function diag_n_dm2(ts::CTMTensors)
    # A = ts.A
    # Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm2(C1, C2, C3, C4, E1, E2, E3, E4)

    n_dm = reshape(n_dm, prod(size(n_dm)[1:6]), :)
    n_dm = (n_dm + n_dm')/2

    vs, vecs = eigen(n_dm)

    vs, vecs
end

function diag_n_dm3(ts::CTMTensors)
    # A = ts.A
    # Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm3(C1, C2, C3, C4, E1, E2, E3, E4)

    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)
    n_dm = (n_dm + n_dm')/2

    vs, vecs = eigen(n_dm)

    vs, vecs
end

function diag_n_dm4(ts::CTMTensors)
    # A = ts.A
    # Ad = ts.Ad
    C1, C2, C3, C4 = ts.Cs
    E1, E2, E3, E4 = ts.Es

    n_dm = get_dm4(C1, C2, C3, C4, E1, E2, E3, E4)

    n_dm = reshape(n_dm, prod(size(n_dm)[1:8]), :)
    n_dm = (n_dm + n_dm')/2

    vs, vecs = eigen(n_dm)

    vs, vecs
end

function get_ts(A)
    H = honeycomb(1, 1)
    cfg = TOML.parsefile("src/default_config.toml")
    ts = iPEPS.CTMTensors(A, cfg);
    conv_fun(_x) = iPEPS.get_gs_energy(_x, H)[1]
    ts, _ = iPEPS.run_ctm(ts, conv_fun = conv_fun);

    ts
end

function get_SvN(ts)
    SvN = zeros(4)
    
    vs = iPEPS.diag_n_dm(ts)[1][vs .> 1e-16] |> x -> sum(p -> -p*log(p), x./sum(x))
    SvN = sum(p -> -p*log(p), vs./sum(vs))
end

A = init_hb_gs(2)
# A = iPEPS.get_Q_ghz()
D = 2
A = randn(ComplexF64, D, D, D, D, 4)
ts = get_ts(A);
iPEPS.diag_n_dm(ts)[1] -> x[x .> 1e-16] |> x -> sum(p -> -p*log(p), x./sum(x))
# get_SvN(ts)

SvN = [iPEPS.diag_n_dm(ts)[1], iPEPS.diag_n_dm2(ts)[1], iPEPS.diag_n_dm3(ts)[1], iPEPS.diag_n_dm4(ts)[1]]

f = x -> x[x .> 1e-16] |> x -> sum(p -> -p*log(p), x./sum(x))
y = SvN .|> f
x = [1, 2, 3, 4]
lines(y)





i = 6
Bi = reshape(basis[:,i] , size(ts.A))
Cs, Es = iPEPS.init_ctm(ts.A, ts.Ad)
ts1 = setproperties(ts, Cs = Cs, Es = Es, B=Bi, Bd=conj(Bi));
ts1, _ = iPEPS.run_ctm(ts1);

bs = iPEPS.get_gauge_basis(ts)