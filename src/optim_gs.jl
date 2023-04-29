using TensorOperations, TensorRules
function optim_GS(H, A0)
    function fg!(F,G,x)
        T = transfer_matrix(x)
        env = get_envtensor(T; chi = chi, output = false)
        f(_x) = get_energy(_x, conj(_x), h_hor, h_ver, env; chi = chi) 
        y, back = Zygote.pullback(f, x)
        if G !== nothing
            copy!(G, back(1)[1])
        end
        if F !== nothing
            return y
        end
    end

    res = optimize(Optim.only_fg!(fg!), A, LBFGS(), Optim.Options(x_tol = 1e-6, f_tol = 1e-6, g_tol = 1e-6))
    res
end

function compute_energy(hh, hv, C1,C2,C3,C4,E1,E2,E3,E4, A)
    dm = density_matrix(A)
    rh = get_rho_hor(C1,C2,C3,C4, E1,E1, E2, E3,E3, E4, dm, dm)
    rv = get_rho_ver(C1,C2,C3,C4, E1, E2,E2, E3, E4,E4, dm, dm)

    rrh = reshape(rh, size(rh,1)*size(rh,2), :)
    hh = reshape(hh, size(hh,1)*size(hh,2), :)
    Eh = tr(hh*rrh)
    Nh = tr(rrh)

    rrv = reshape(rv, size(rv,1)*size(rv,2), :)
    hv = reshape(hv, size(hv,1)*size(hv,2), :)
    Ev = tr(hv*rrv)
    Nv = tr(rrv)

    E = Eh/Nh + Ev/Nv
    E
end

transfer_matrix(A::AbstractArray) = transfer_matrix(A, conj(A)) 

@∇ function transfer_matrix(A::AbstractArray,B::AbstractArray)
    @tensor T[a1,a2,b1,b2,c1,c2,d1,d2] := A[i, a1, b1, c1, d1]*B[i, a2, b2, c2, d2]
    dim = size(T)
    T = reshape(T, dim[1]*dim[2], dim[3]*dim[4], dim[5]*dim[6], dim[7]*dim[8])
    T
end

"""
    density_matrix

Return a tensor(d, d, D²,D²,D²,D²)
"""
density_matrix(A::AbstractArray) = density_matrix(A, conj(A)) 

@∇ function density_matrix(A::AbstractArray,B::AbstractArray)
    @tensor T[i,j, a1,a2,b1,b2,c1,c2,d1,d2] := A[i, a1, b1, c1, d1]*B[j, a2, b2, c2, d2]
    dim = size(T)
    T = reshape(T, dim[1], dim[2], dim[3]*dim[4], dim[5]*dim[6], dim[7]*dim[8], dim[9]*dim[10])
    T
end

    """
    get_dm_hor

return density matrix(d,d,d,d) of following diagram
```
C1 -- E1l -- E1r -- C2
|     |      |      |
E4 -- dml<  >dmr -- E2  = ρₕ
|     |      |      |  
C4 -- E3l -- E3r -- C3
```
"""
@∇ function get_rho_hor(C1, C2, C3, C4, E1l, E1r, E2, E3l, E3r, E4, dml, dmr)
    # left
    @tensor ul[i,j, m1,m2,m3,m4] := C1[p2,p1]*E1l[p1,p3,m3]*E4[p2,m1,p4]*dml[i,j, p3,p4,m2,m4]
    @tensor bl[m1,m2,m3] := C4[m1, p1]*E3l[m2, p1, m3]
    @tensor rhol[i,j, m1,m2,m3] := ul[i,j, p1,p2, m1,m2]*bl[p1,p2, m3]
    # right
    @tensor ur[i,j, m1,m2,m3,m4] := C2[p1,p2]*E1r[m1, p3,p1]*E2[p2,p4, m3]*dmr[i,j, p3,m2,m4,p4]
    @tensor br[m1,m2,m3] := E3r[m2,m3, p1]*C3[m1, p1]
    @tensor rhor[i,j, m1,m2,m3] := ur[i,j, m1,m2, p1,p2]*br[p1,p2, m3]

    @tensor rho[m1,m2,m3,m4] := rhol[m1,m3, p1,p2,p3]*rhor[m2,m4, p1,p2,p3]

    rho
end

"""
    get_dm_ver

return density matrix(d,d,d,d) of following diagram
```
C1 --  E1 -- C2
|      |     |   
E4u -- dmu-- E2u
|      Λ     |  =  ρᵥ
|      V     |
E4d -- dmd-- E2d  
|      |     |  
C4 --  E3 -- C3
```
"""
@∇ function get_rho_ver(C1, C2, C3, C4, E1, E2u, E2d, E3, E4u, E4d, dmu, dmd)
    #up
    @tensor ul[i,j, m1,m2,m3,m4] := C1[p2,p1]*E1[p1,p3, m3]*E4u[p2, m1, p4]*dmu[i,j, p3,p4, m2,m4]
    @tensor ur[m1,m2,m3] := C2[m1, p1]*E2u[p1, m2,m3]
    @tensor rhou[i,j, m1,m2,m3] := ul[i,j, m1,m2, p1,p2]*ur[p1,p2, m3]
    #bottom
    @tensor dl[i,j, m1,m2,m3,m4] := C4[p2,p1]*E3[p4,p1, m3]*E4d[m1, p2,p3]*dmd[i,j, m2, p3,p4, m4]
    @tensor dr[m1,m2,m3] := E2d[m3,m2, p1]*C3[p1, m1]
    @tensor rhod[i,j, m1,m2,m3] := dl[i,j, m1,m2, p1,p2]*dr[p1,p2, m3]

    @tensor rho[m1,m2,m3,m4] := rhou[m1,m3, p1,p2,p3]*rhod[m2,m4, p1,p2,p3] 

    rho
end