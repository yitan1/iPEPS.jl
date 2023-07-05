using Random

include("pepo.jl")
include("doubleTensors.jl")
include("eval_con.jl")
include("variationalAs.jl")
function main()
    τ=0.01; pepo =PEPOTensors(init_pepo(τ)...)    
    D,s,χ=4,2,50
    rng=MersenneTwister(1234);
    As = randn(rng,Float64,(D,D,D,D,s)); As /= norm(As)
    en=eval_en(pepo,As)
    
    for i=1:100
        As=opt_As(pepo,As); As /= norm(As)
        # @show en0=en
        
        # @show en=eval_en(pepo,As1)
        # if en<=en0
            # As .= As1
        # end
    end
    return nothing
    # dtn=DoubleTensors(init_DoubleTensor(As)...)        

    # dth=DoubleTensors(init_DoubleTensor(As,pepo)...)

    # dth,dtn=envNormalize(dth,dtn)    
    # CTh,CTn=envCT(dth),envCT(dtn)

    # energy= x->eval_en(CTh,CTn,pepo,x)
    # ∂energy=x->eval_∂E(CTh,CTn,pepo,x)

    # optimargs = (Optim.Options(show_trace=true,iterations=1000),); optimmethod = LBFGS(m = 20);
    # @show result = optimize(energy,∂energy,As,optimmethod,inplace=false,optimargs...)
    # @show size(result.minimizer)



    return nothing

end

main()
