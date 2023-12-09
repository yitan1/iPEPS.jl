
@∇ function op_expect(A ,C, E, op) # op is two-sites operotor and should be matrix
    d = size(A,1)
    D = size(A,2)
    @tensor T[a,e,b,f,c,g,d,h, i,j] := A[i, a,b,c,d]*conj(A)[j, e,f,g,h]
    Td = reshape(T, D*D, D*D, D*D, D*D, d,d)

    @tensor CE[a,b,c] := C[i,a]*E[i,b,c] 
    @tensor EL[a,b,c,d] := CE[i,c,d]*E[i,a,b]
    @tensor EL[a,b,c,d, e,f] := EL[i,a, j,c]*Td[i,j, d,b, e,f]
    @tensor EL[a,b,c, e,f] := EL[a,b,i,j, e,f]*CE[i,j, c]
    @tensor rho[a,c,b,d] := EL[i,j,k, a,b]*EL[i,j,k, c,d]

    rho = reshape(rho, d*d, d*d)
    rho = 0.5*(rho + rho')

    norm_rho = tr(rho)

    energy = tr(rho*op)/norm_rho

    energy
end


@∇ function op4_expect(A ,C, E, op) # op is four-sites operotor and should be matrix
    d = size(A,1)
    D = size(A,2)
    @tensor T[a,e,b,f,c,g,d,h, i,j] := A[i, a,b,c,d]*conj(A)[j, e,f,g,h]
    Td = reshape(T, D*D, D*D, D*D, D*D, d,d)

    @tensor CE[a,b,c] := C[i,a]*E[i,b,c] 
    @tensor EL[a,b,c,d] := CE[i,c,d]*E[i,a,b]
    @tensor EL[a,b,c,d, e,f] := EL[i,a, j,c]*Td[i,j, d,b, e,f]
    @tensor EL[a,b,c,d, e,f,x,y] := EL[a,b,i,j, e,f]*EL[c,d,i,j, x,y]
    @tensor rho[a,c,w,y,b,d,x,z] := EL[i,j,k,m, a,b,c,d]*EL[i,j,k,m, w,x,y,z]

    rho = reshape(rho, d*d*d*d, d*d*d*d)
    rho = 0.5*(rho + rho')

    norm_rho = tr(rho)

    energy = tr(rho*op)/norm_rho

    energy
end