"""
    get_hor_energy

```
C1 -- E1 -- E1 -- C2
|     |     |     |
E4 -- T -h- T  -- E2  = E_hor
|     |     |     |  
C4 -- E3 -- E3 -- C3
```
"""
function get_hor_energy(h, envs::ExcEnvTensor, phi1::ExcIPEPS, phi2::ExcIPEPS)
    Cs, Es = corner(envs[1]), edge(envs[1]), bulk
    Cs_B, Es_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])
    Cs_Bd, Es_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])
    Cs_BB, Es_BB = corner(envs[4]), edge(envs[4]), bulk(envs[4])

    #get out-product States
    A_x_Ad = density_matrix(get_A(phi1), get_Ad(phi2))
    B_x_Ad = density_matrix(get_B(phi1), get_Ad(phi2))
    A_x_Bd = density_matrix(get_A(phi1), get_Bd(phi2))
    B_x_Bd = density_matrix(get_B(phi1), get_Bd(phi2))

    kx ,ky = get_kx(phi1), get_ky(phi1)

    W1 = [Cs[1], Es[1], Es[1], Cs[2], Es[4], A_x_Ad, A_x_Ad, Es[2], Cs[4], Es[3], Es[3], Cs[3]]
    W2 = [Cs_B[1], Es_B[1], Es_B[1], Cs_B[2], Es_B[4], B_x_Ad, B_x_Ad, Es_B[2], Cs_B[4], Es_B[3], Es_B[3], Cs_B[3]]
    W3 = [Cs_Bd[1], Es_Bd[1], Es_Bd[1], Cs_Bd[2], Es_Bd[4], A_x_Bd, A_x_Bd, Es_Bd[2], Cs_Bd[4], Es_Bd[3], Es_Bd[3], Cs_Bd[3]]
    W4 = [Cs_BB[1], Es_BB[1], Es_BB[1], Cs_BB[2], Es_BB[4], B_x_Bd, B_x_Bd, Es_BB[2], Cs_BB[4], Es_BB[3], Es_Bd[3], Cs_Bd[3]]
    
    E_hor = 0.0
    for i in 1:12
        for j in 1:12
            W = get_W([W1, W2, W3, W4], i, j)
            dx = (j-1) % 4 - (i-1) % 4
            dy = (j-1) ÷ 4 - (i-1) ÷ 4
            if dx == 1
                dx = 0
            end
            if dy == 1
                dy = 0
            end
            a = exp(-im*kx*dx)*exp(-im*ky*dy)
            
            effE1 = contract_E1(W[2], W[3])
            effE3 = contract_E3(W[10], W[11])
            effT = contract_hor_T(h, W[6], W[7])
            E_hor += contract_env(W[1], W[4], W[12], W[9], effE1, W[8], effE3, W[5], effT)*a
        end
    end
    E_hor = E_hor/144
    @show E_hor
    real(E_hor)
end

#### XXX ugly
function get_W(Ws, i, j)
    W1, W2, W3, W4 = undef, undef, undef, undef
    W5, W6, W7, W8 = undef, undef, undef, undef
    W9, W10, W11, W12 = undef, undef, undef, undef
    for k = 1:12
        if k == i && i == j
            n = 4
        elseif k == i 
            n = 2
        elseif k == j
            n = 3
        else 
            n = 1
        end

        if  k == 1
            W1 = Ws[n][k]
        end
        if  k == 2
            W2 = Ws[n][k]
        end
        if k == 3
            W3 = Ws[n][k]
        end
        if k == 4
            W4 = Ws[n][k]
        end
        if k == 5
            W5 = Ws[n][k]
        end
        if k == 6
            W6 = Ws[n][k]
        end
        if k == 7
            W7 = Ws[n][k]
        end
        if k == 8
            W8 = Ws[n][k]
        end
        if k == 9
            W9 = Ws[n][k]
        end
        if k == 10
            W10 = Ws[n][k]
        end
        if k == 11
            W11 = Ws[n][k]
        end
        if k == 12
            W12 = Ws[n][k]
        end
    end

    W = [W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12]
    W
end

"""
    get_ver_energy

```
C1 -- E1 -- C2
|     |     |   
E4 -- T  -- E2
|     h     |  =  E_ver
E4 -- T  -- E2  
|     |     |  
C4 -- E3 -- C3
```
"""
function get_ver_energy(h, envs::ExcEnvTensor, phi1::ExcIPEPS, phi2::ExcIPEPS) #TODO 
    Cs, Es = corner(envs[1]), edge(envs[1]), bulk
    Cs_B, Es_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])
    Cs_Bd, Es_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])
    Cs_BB, Es_BB = corner(envs[4]), edge(envs[4]), bulk(envs[4])

    #get out-product States
    A_x_Ad = density_matrix(get_A(phi1), get_Ad(phi2))
    B_x_Ad = density_matrix(get_B(phi1), get_Ad(phi2))
    A_x_Bd = density_matrix(get_A(phi1), get_Bd(phi2))
    B_x_Bd = density_matrix(get_B(phi1), get_Bd(phi2))

    kx ,ky = get_kx(phi1), get_ky(phi1)


    W1 = [Cs[1], Es[1], Cs[2], Es[4], A_x_Ad, Es[2], Es[4], A_x_Ad, Es[2], Cs[4], Es[3], Cs[3]]
    W2 = [Cs_B[1], Es_B[1], Cs_B[2], Es_B[4], B_x_Ad, Es_B[2], Es_B[4], B_x_Ad, Es_B[2], Cs_B[4], Es_B[3], Cs_B[3]]
    W3 = [Cs_Bd[1], Es_Bd[1], Cs_Bd[2], Es_Bd[4], A_x_Bd, Es_Bd[2], Es_Bd[4], A_x_Bd, Es_Bd[2], Cs_Bd[4], Es_Bd[3], Cs_Bd[3]]
    W4 = [Cs_BB[1], Es_BB[1], Cs_BB[2], Es_BB[4], B_x_Bd, Es_BB[2], Es_BB[4], B_x_Bd, Es_BB[2], Cs_BB[4], Es_BB[3], Cs_Bd[3]]

    E_ver = 0.0
    for i in 1:12
        for j in 1:12
            W = get_W([W1, W2, W3, W4], i, j)

            dx = (j-1) % 3 - (i-1) % 3
            dy = (j-1) ÷ 3 - (i-1) ÷ 3
            if dx == 1
                dx = 0
            end
            if dy == 1
                dy = 0
            end
            a = exp(-im*kx*dx)*exp(-im*ky*dy)
            
            effE2 = contract_E2(W[6], W[9])
            effE4 = contract_E4(W[4], W[7])
            effT = contract_ver_T(h, W[5], W[8])
            E_ver += contract_env(W[1], W[3], W[12], W[10], W[2], effE2, W[11], effE4, effT)*a
        end
    end
    E_ver = E_ver/144
    @show E_ver
    real(E_ver)
end

"""
    get_norm

```
C1 -- E1 -- C2
|     |     |     
E4 -- T  -- E2  = N
|     |     |  
C4 -- E3 -- C3
```
"""
function get_norm(env::EnvTensor)
    T = bulk(env)
    get_norm(env, T)
end

function get_norm(env::EnvTensor, T)
    Cs = corner(env)
    Es = edge(env)

    contract_env(Cs[1], Cs[2], Cs[3], Cs[4], Es[1], Es[2], Es[3], Es[4], T)
end