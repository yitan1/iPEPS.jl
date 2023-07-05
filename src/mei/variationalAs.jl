include("ctmrg.jl")
using Optim, LineSearches
function opt_As(pepo,As)
    dtn=DoubleTensors(init_DoubleTensor(As)...)        

    dth=DoubleTensors(init_DoubleTensor(As,pepo)...)
    χ=50

    for i=1:1
        dth,dtn=envNormalize(dth,dtn)    
        dth,vals=ctmrgstep(dth;χ=χ)
        dtn,valsn=ctmrgstep(dtn;χ=χ)
        @show vals[1:10]/vals[1], valsn[1:10]/valsn[1]
    end
    dth,dtn=envNormalize(dth,dtn)    

    CTh,CTn=envCT(dth),envCT(dtn)

    energy= x->eval_en(CTh,CTn,pepo,x)
    ∂energy=x->eval_∂E(CTh,CTn,pepo,x)

    optimargs = (Optim.Options(show_trace=true,iterations=5),); optimmethod = LBFGS(m = 20);
    @show result = optimize(energy,∂energy,As,optimmethod,inplace=false,optimargs...)
    # @show result.minimum
    return result.minimizer
end