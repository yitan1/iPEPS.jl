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
    for i = 1:12
        for j = 1:12
    # contract_env(C1, C2, C3, C4, effE1, E2, effE3, E4, effT)
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