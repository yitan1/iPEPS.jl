# TODO

function get_vison(D; p1=0.24, p2=0.0)
    Q_op = zeros(ComplexF64,2,2,2,2,2)
    Q_op[1,1,1,:,:] = SI
    Q_op[1,2,2,:,:] = sigmax
    Q_op[2,1,2,:,:] = sigmay
    Q_op[2,2,1,:,:] = sigmaz

    ux, uy, uz = 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)
    s111 = 1/sqrt(2+2*uz)*[1 + uz, ux + im*uy]

    T = tcon([Q_op, s111], [[-1,-2,-3,-4,1], [1]])

    @ein T1[m1,m2,m3,m4] := T[p1,m2,m3,m4]*sigmaz[p1,m1]
    # A = tcon([T1, T], [[-1,-2,1,-5], [-3,-4,1,-6]]) # ZZ
    A = tcon([T1, T], [[1,-1,-2,-5], [1,-3,-4,-6]]) # XX
    A = reshape(A, 2, 2, 2, 2, 4)

    if D == 4
        phi = p1 * pi
        theta = exp(-im * pi * p2)
        R_op = zeros(ComplexF64, 2, 2, 2, 2, 2)
        R_op[1, 1, 1, :, :] = SI .* cos(phi)
        R_op[2, 1, 1, :, :] = 2 * Sx * sin(phi) * theta
        R_op[1, 2, 1, :, :] = 2 * Sy * sin(phi) * theta
        R_op[1, 1, 2, :, :] = 2 * Sz * sin(phi) * theta

        RR = tcon([R_op, R_op], [[-1, -2, 1, -5, -7], [-3, -4, 1, -6, -8]])
        dRR = size(RR)
        RR = reshape(RR, dRR[1], dRR[2], dRR[3], dRR[4], dRR[5] * dRR[6], dRR[7] * dRR[8])

        A = tcon([RR, A], [[-1, -3, -5, -7, -9, 1], [-2, -4, -6, -8, 1]])
        D1 = size(A, 1)
        D2 = size(A, 2)
        A = reshape(A, D1 * D2, D1 * D2, D1 * D2, D1 * D2, size(A, 9))
    end

    A
end


function optim_es_noad(px, py, cfg)
    basis_name = get_basis_name(cfg)
    basis = load(basis_name, "basis")
    ts = load(basis_name, "ts")
    ts = setproperties(ts, Params = cfg)
    H = load(basis_name, "H")
    fprint("load basis, ts, H in $basis_name")

    basis = complex(basis) # ！！！！ convert Complex
    basis_dim = size(basis, 2) 

    ts.Params["px"] = convert(eltype(ts.A), px*pi)
    ts.Params["py"] = convert(eltype(ts.A), py*pi)
    es_name = get_es_name(ts.Params, px, py)
    if  ts.Params["es_resume"] > 0 && ispath(es_name) 
        es_file = load(es_name)
        effH = es_file["effH"]
        effN = es_file["effN"]
        if haskey(es_file, "envB")
            envB = load(es_name, "envB")
            println("load envB")
        else
            envB = zeros(ComplexF64, size(basis))
            println("new envB")
        end
        fprint("load existed calculation , effH, effN in $es_name")
    else
        effH = zeros(ComplexF64, basis_dim, basis_dim)
        effN = zeros(ComplexF64, basis_dim, basis_dim)
        envB = zeros(ComplexF64, size(basis))
    end

    for i = 1:basis_dim
        if i < ts.Params["es_resume"]
            fprint(" Simulation of basis vector $(i)/$(basis_dim) existed, skip to next")
            continue
        end
        if ts.Params["es_num"] > 0 && i >= (ts.Params["es_resume"] + ts.Params["es_num"])
            fprint("\nUp to maximum simulation of basis vector $(i)/$(basis_dim) existed, end to calculation")
            break
        end
        fprint(" \n Starting simulation of basis vector $(i)/$(basis_dim)")
        for j = 1:basis_dim
            fprint(" \n Starting simulation of basis vector $(j)/$(i)")
            Hij, Nij = get_es_element(ts, H, basis[:,i], basis[:,j])
            effH[j, i] = Hij
            effN[j, i] = Nij
        end
        fprint("Finish basis vector of $(i)/$(basis_dim)")
        jldsave(es_name; effH = effH, effN = effN)
    end
    
    jldsave(es_name; effH = effH, effN = effN)
    fprint("Saved (effH, effN) to $(es_name)")

    effH, effN
end

function get_es_element(ts, H, Bi, Bj)
    Bi = reshape(Bi, size(ts.A))
    Bj = reshape(Bj, size(ts.A))

    # Cs, Es = init_ctm(ts.A, ts.Ad)

    ts1 = setproperties(ts, B = Bj, Bd = conj(Bi))

    fprint("\n ---- Start to find fixed points -----")
    conv_fun(_x) = get_es_energy(_x, H) / get_all_norm(_x)[1]
    ts1, _ = run_ctm(ts1, conv_fun = conv_fun)
    fprint("---- End to find fixed points ----- \n")

    max_iter = ts1.Params["max_iter"]
    ts1.Params["max_iter"] = ts1.Params["ad_max_iter"]

    ts1, _ = run_ctm(ts1, conv_fun = conv_fun)
    y = get_es_energy(ts1, H)

    ts1.Params["max_iter"] = max_iter
    
    Nb, _ = get_all_norm(ts1)
    fprint("Energy: $y \nNormB: $(Nb) ")
    
    y, Nb
end 

function optim_es1(ts::CTMTensors, H, px, py)
        ## normalize gs
        ts1 = normalize_gs(ts)

        H = substract_gs_energy(ts1, H)
    
        #basis
        basis_name = get_basis_name(ts1.Params)
        # post = ".jld2"
        if ispath(basis_name) && ts1.Params["basis"]
            basis = load(basis_name, "basis")
            fprint("The basis has existed, skip calculation")
        else
            basis = get_tangent_basis(ts)
            jldsave(basis_name; basis = basis)
            fprint("Saved the basis to $(basis_name)")
        end
    
        basis = complex(basis) # ！！！！ convert Complex
        basis_dim = size(basis, 2) 
        effH = zeros(ComplexF64, basis_dim, basis_dim)
        effN = zeros(ComplexF64, basis_dim, basis_dim)
    
        ts.Params["px"] = convert(eltype(ts.A), px*pi)
        ts.Params["py"] = convert(eltype(ts.A), py*pi)
    
        f = x -> get_es_grad(ts1, H, x)
        
        vals , vec = geneigsolve(f, 32, 3, :LM, ishermitian = true)
end