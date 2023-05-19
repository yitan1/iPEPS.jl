
using iPEPS
# using MKL

H = ising();

A = randn(2,2,2,2,2);

res = optim_gs(H, A, "")

px, py = make_es_path()

# for i in eachindex(px)
    optim_es(H, px[1], py[1], "")
# end   


