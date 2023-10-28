using Test
using iPEPS
using iPEPS
using OMEinsum
using TOML
using ConstructionBase

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