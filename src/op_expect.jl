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
    E_hor = 0.0
    Cs, Es = corner(envs[1]), edge(envs[1]), bulk
    Cs_B, Es_B = corner(envs[2]), edge(envs[2]), bulk(envs[2])
    Cs_Bd, Es_Bd = corner(envs[3]), edge(envs[3]), bulk(envs[3])
    Cs_BB, Es_BB = corner(envs[4]), edge(envs[4]), bulk(envs[4])

    #get out-product States
    A_x_Ad = density_matrix(get_A(phi1), get_Ad(phi2))
    B_x_Ad = density_matrix(get_B(phi1), get_Ad(phi2))
    A_x_Bd = density_matrix(get_A(phi1), get_Bd(phi2))
    B_x_Bd = density_matrix(get_B(phi1), get_Bd(phi2))


    W1 = [Cs[1], Es[1], Es[1], Cs[2], Es[4], A_x_Ad, A_x_Ad, Es[2], Cs[4], Es[3], Es[3], Cs[3]]
    W2 = [Cs_B[1], Es_B[1], Es_B[1], Cs_B[2], Es_B[4], B_x_Ad, B_x_Ad, Es_B[2], Cs_B[4], Es_B[3], Es_B[3], Cs_B[3]]
    W3 = [Cs_Bd[1], Es_Bd[1], Es_Bd[1], Cs_Bd[2], Es_Bd[4], A_x_Bd, A_x_Bd, Es_Bd[2], Cs_Bd[4], Es_Bd[3], Es_Bd[3], Cs_Bd[3]]
    W4 = [Cs_BB[1], Es_BB[1], Es_BB[1], Cs_BB[2], Es_BB[4], B_x_Bd, B_x_Bd, Es_BB[2], Cs_BB[4], Es_BB[3], Es_Bd[3], Cs_Bd[3]]

    W = copy(W1) # XXX ugly 
    for i in eachindex(W)
        for j in eachindex(W)
            if i == j
                W[i] = W4[i]
            else 
                W[i] = W2[i]
                W[j] = W3[j]
            end
            
            effE1 = contract_E1(W[2], W[3])
            effE3 = contract_E3(W[10], W[11])
            effT = contract_hor_T(h, W[6], W[7]) 
            E_hor += contract_env(W[1], W[4], W[12], W[9], effE1, W[8], effE3, W[5], effT)
        end
    end
    E_hor
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
function get_ver_energy(h, envs::ExcEnvTensor, phi1::ExcIPEPS, phi2::ExcIPEPS)
    E_ver = 0.0
    for i = 1:12
        for j = 1:12
    # contract_env(C1, C2, C3, C4, effE1, E2, effE3, E4, effT)
        end
    end
    E_ver
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