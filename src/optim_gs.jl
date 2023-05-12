
function optim_GS(H, A0, chi)
    energies = Float64[]
    gradnorms = Float64[]

    cached_x = nothing
    cached_y = nothing 
    cached_g = nothing

    function verbose(xk)
        # println(xk)
        if cached_y !== nothing && cached_g !== nothing 
            append!(energies, cached_y)
            append!(gradnorms, norm(cached_g))
        end
        println(" # ======================== #")
        println(" #      Step completed      #")
        println(" # ======================== #")
        [@printf(" Step %3d  E: %0.8f  |grad|: %0.8f \n", i, E, gradnorms[i]) for (i, E) in enumerate(energies)]

        return false
    end

    function fg!(F,G,x)
        x = renormalize(x)

        # if cached_g !== nothing && cached_x !== nothing && norm(x - cached_x) < 1e-14
        #     println("Restart to find x")
        #     if G !== nothing
        #         copy!(G, cached_g)
        #     end
        #     if F !== nothing
        #         return cached_y
        #     end
        # end

        ts0 = CTMTensors(x, x);
        conv_fun(_x) = get_gs_energy(H, _x)
        println("\n ---- Start to find fixed points -----")
        ts0, _ = run_ctm(ts0, chi; conv_fun = conv_fun);
        println("---- End to find fixed points ----- \n")
        f(_x) = run_energy(H, ts0, chi, _x) 
        @time y, back = Zygote.pullback(f, x)

        println("Finish autodiff")
        cached_x = x
        cached_y = y

        if G !== nothing
            @time g = back(1)[1]
            cached_g = g
            # @show g
            copy!(G, g)
        end
        if F !== nothing
            @printf("Gs_Energy: %.10g \n", y)
            return y
        end
    end

    # optimizer = L_BFGS_B(1024, 17)
    # res = optimizer(Optim.only_fg!(fg!), A0, m=20, factr=1e7, pgtol=1e-5, iprint=-1, maxfun=15000, maxiter=15000)

    res = optimize(Optim.only_fg!(fg!), A0, LBFGS(), Optim.Options( g_tol=1e-6, callback = verbose, iterations = 2))

    res
end

function run_energy(H, ts0, chi, A)
    # ts0 = iPEPS.CTMTensors(A,A)
    ts0 = setproperties(ts0, A = A, Ad = conj(A))

    conv_fun(_x) = get_gs_energy(H, _x)
    ts, s = run_ctm(ts0, chi, conv_fun = conv_fun)
    
    # ts, s = iPEPS.run_ctm(conv_ts, 50)
    E, N = get_energy(H, ts)

    # gs_E = (E[1]/N[1] + E[2]/N[2])/2
    # @printf("Gs_Energy: %.10g \n", sum(E))
    # println("E: $E, N: $N")

    E[1] + E[2]
end

function get_gs_energy(H, ts)
    E, _ = get_energy(H, ts)
    E[1] + E[2]
end

function get_energy(H, ts)
    A = ts.A 
    Ad = ts.Ad
    roh, rov = get_dms(ts)
    Nh = tr(roh)
    Nv = tr(rov)
    roh = roh ./ Nh
    rov = rov ./ Nv
    Eh = tr(H[1]*roh)
    Ev = tr(H[2]*rov)
    E = [Eh, Ev]
    N = [Nh, Nv]

    E, N
end

function get_dms(ts; only_gs = true)
    if only_gs
        A    = ts.A
        Ad   = ts.Ad
        C1   = ts.Cs[1]
        C2   = ts.Cs[2]
        C3   = ts.Cs[3]
        C4   = ts.Cs[4]
        E1   = ts.Es[1]
        E2   = ts.Es[2]
        E3   = ts.Es[3]
        E4   = ts.Es[4]
    else
        A    = get_all_A(ts)
        Ad   = get_all_Ad(ts)
        C1, C2, C3, C4 = get_all_Cs(ts)
        E1, E2, E3, E4 = get_all_Es(ts)
    end

    ts_h = [C1, C2, C3, C4, E1, E1, E2, E3, E3, E4, A, A, Ad, Ad]
    ts_v = [C1, C2, C3, C4, E1, E2, E2, E3, E4, E4, A, A, Ad, Ad]
    roh = get_dm_h(ts_h...)
    rov = get_dm_v(ts_v...)

    return roh, rov
end
"""
    get_dm_hor

return density matrix(d,d,d,d) of following diagram
```
C1 -- E1l -- E1r -- C2
|     ||     ||      |
E4 == AAl< =>AAr == E2  = ρₕ
|     ||     ||      |  
C4 -- E3l -- E3r -- C3
```
"""
function get_dm_h(C1, C2, C3, C4, E1l, E1r, E2, E3l, E3r, E4, Al, Ar, Adl, Adr)
    #left
    LU = begin
        C1E1 = tcon([C1, E1l], [[-1,1], [1, -2, -3, -4]])
        tcon([C1E1, Adl], [[-1,-2, -3, 1], [1, -4,-5,-6,-7]])
    end
    LB = begin
        C4E3 = tcon([C4, E3l], [[-1,1], [1,-2, -3, -4]])    # [t, r], [l, r, bra, ket] -> [t, r, bra, ket]
        CEE4 = tcon([E4, C4E3], [[-1,1, -3,-5], [1, -2, -4, -6]])  # [t, b, bra, ket], [t, r, bra, ket] -> 
        tcon([CEE4, Al], [[-1,-2,1,2,-4,-5], [-3, 1, 2, -6, -7]])  
    end  
    L = tcon([LB,LU], [[1,-2,2,3,4,-3,-5], [1,-1,2,3,4,-4,-6]])

    #right
    RU = begin
        C2E1 = tcon([C2, E1r], [[1,-2],[-1, 1, -3, -4]])
        tcon([C2E1, Ar], [[-1, -2, 1, -3], [1, -4, -5, -6, -7]])
    end 
    RB = begin
        C3E3 = tcon([C3, E3r], [[-1,1],[-2,1, -3,-4]])
        CEE2 = tcon([C3E3, E2], [[1,-2, -3, -5],[-1,1, -4,-6]])
        tcon([CEE2, Adr], [[-1,-2,-3,-4,1,2], [-5,-6,1,2,-7]])
    end
    R = tcon([RU,RB], [[-1,1,2,-3,3,4,-5],[1,-2,3,4,2,-4,-6]])

    roh = tcon([L,R],[[1,2,3,4,-1,-3],[1,2,3,4,-2,-4]])

    d = size(Al, 5)
    roh = reshape(roh, d*d, d*d)

    return roh
end

"""
    get_dm_v

return density matrix(d,d,d,d) of following diagram
```
C1 --  E1 -- C2
|      |     |   
E4u -- Au-- E2u
|      Λ     |  =  ρᵥ
|      V     |
E4d -- Ad-- E2d  
|      |     |  
C4 --  E3 -- C3
```
"""
function get_dm_v(C1, C2, C3, C4, E1, E2u, E2d, E3, E4u, E4d, Au, Ad, Adu, Add)
    #up
    UL = begin
        C1E1 = tcon([C1, E1], [[-1,1], [1,-2,-3,-4]])
        CEE4 = tcon([C1E1, E4u], [[1,-2,-3,-5], [1,-1,-4,-6]])
        tcon([CEE4, Au], [[-1,-2,1,2,-3,-4], [1,2,-5,-6,-7]])
    end
    UR = begin
        C2E2 = tcon([C2,E2u], [[-1,1], [1,-2,-3,-4]])
        tcon([C2E2, Adu], [[-1,-2,-3,1],[-4,-5,-6,1,-7]])
    end
    U = tcon([UL,UR], [[-1,1,3,4,-3,2,-5], [1,-2,2,3,4,-4,-6]])

    #bottom
    BL = begin
        C4E3 = tcon([C4,E3], [[-1,1], [1,-2,-3,-4]])
        CEE4 = tcon([C4E3, E4d], [[1,-2,-4,-6],[-1,1,-3,-5]])
        tcon([CEE4, Add], [[-1,-2,-3,-4,1,2], [-5,1,2,-6,-7]])
    end
    BR = begin
        C3E2 = tcon([C3, E2d], [[1,-2], [-1,1,-3,-4]])
        tcon([C3E2, Ad], [[-1,-2,1,-3], [-4,-5,-6,1,-7]])
    end
    B = tcon([BL, BR], [[-1,1,2,3,-4,4,-6], [-2,1,4,-3,2,3,-5]])

    rov = tcon([U,B], [[1,2,3,4,-1,-3], [1,2,3,4,-2,-4]])

    d = size(Au, 5)
    rov = reshape(rov, d*d, d*d)

    return rov
end