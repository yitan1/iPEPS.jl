# iPEPS

This is a julia package about basic implementation of iPEPS, including optimization of the ground-state and excited-state. We use CTMRG contraction algorithm, and use autodiff to solve the gradient. More details refer to the  `SciPost Phys. 12, 006 (2022)`

## Installation:

To install, type `]` in a julia (>=1.5) REPL and then input

```
pkg> add https://github.com/yitan1/iPEPS.jl.git
```

## Example
First, we need to optimize the ground state:
```julia
# example_gs.jl

    using iPEPS

    # H = [Hh, Hv],  Hh = -Sx*Sx - h*Sz, default： h =2
    H = ising() 

    # bond dimension D
    D = 2    

    # physical dimension d matching the Hamiltonian dimension
    d = 2  # for spin 1/2

    # initial states
    A = randn(D,D,D,D,d)

    # optim ground state
    res = optim_gs(H, A, "config.toml")
```

Run the script, then it will generate file to directory `./similation/***/gs.jld2`, where `***` is dependent to `config.toml` as following.

Next we will use the ground state to optimize excited state:

```julia
#example_es.jl
using iPEPS

H = ising()
# construct momentum path
# default: M(pi,0) -> X(pi,pi) -> S(pi/2,pi/2) -> Gamma(0,0) -> M(pi,0) -> S(pi/2,pi/2)
px, py = make_es_path()

for i in eachindex(px)
    println(" \n------------ Start simulation : px = $(px[i]), py = $(py[i]) --------\n ")
    optim_es(H, px[i], py[i], "config.toml")
    println(" \n------------ End simulation : px = $(px[i]), py = $(py[i]) -------- \n ")
end
```
After above long computation, we can plot the excitation band:
```julia
# example_plot.jl
using iPEPS
using CarioMakie

# number of bands
n = 4
E = plot_band(n, "config.toml")

f = Figure()
ax = Axis(f[1, 1])
for i = 1:n
    x = collect(1:size(E,2))
    y = E[i,:]
    lines!(ax, x, y)
end
f
```

#### Appendix
```toml
# config.toml
out_prefix = "default"
model = "ising"

D = 2
chi = 30

resume = false

nrmB_cut = 1e-3

min_iter = 4
max_iter = 30
rg_tol = 1e-6
```

## Package Structure:
- iPEPS.jl: main file


- Basis file: 
          
    1. printing.jl: custom output
    2. io.jl: create dir and file 
    3. tcon.jl: wrap the contraction function
    4. model.jl: construct Hamiltonian
    5. svd_ad.jl: svd for zygote extend
    6. basis.jl: 

- Struct file: 

    1. emptyT.jl: convenient struct for contraction 
    2. nested_tensor.jl: tensor with [T, T_B, T_Bd, T_Bd]
    3. ctm_tensor.jl: Struct of Corner transfer matrix

- Optimization file:

    1. ctmrg.jl: update ctm_tensor
    2. optim_gs.jl: optimize the ground state 
    3. optim_es.jl: optimize the excited state
    4. expectation.jl: compute the expectation

- Others:

    1. plot.jl


