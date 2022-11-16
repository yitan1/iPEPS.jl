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