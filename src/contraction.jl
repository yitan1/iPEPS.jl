"""
    transfer_matrix

Return a tensor(D²,D²,D²,D²)
"""
transfer_matrix(A::AbstractArray) = transfer_matrix(A, conj(A)) 

@∇ function transfer_matrix(A::AbstractArray,B::AbstractArray)
    @tensor T[a1,a2,b1,b2,c1,c2,d1,d2] := A[i, a1, b1, c1, d1]*B[i, a2, b2, c2, d2]
    dim = size(T)
    T = reshape(T, dim[1]*dim[2], dim[3]*dim[4], dim[5]*dim[6], dim[7]*dim[8])
    T
end

"""
    transfer_matrix

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
```
-- E1a -- E1b --
   |      |
```
"""
@∇ function contract_E1(E1a, E1b)
    @tensor out[m1,m2,m3,m4] := E1a[m1,m2, p1]*E1b[p1, m3, m4]
    out = reshape(out, size(out,1), :, size(out,4))
    out
end

"""
```
   |
-- E2a
   |
-- E2b
   |
```
"""
@∇ function contract_E2(E2a, E2b)
    @tensor out[m1,m2,m3,m4] := E2a[m1,m2, p1]*E2b[p1, m3, m4]
    out = reshape(out, size(out,1), :, size(out,4))
    out
end

"""
```
   |      | 
-- E3a -- E3b --
```
"""
@∇ function contract_E3(E3a, E3b)
    @tensor out[m1,m2,m3,m4] := E3a[m1,m3, p1]*E3b[m2, p1, m4]
    out = reshape(out, :, size(out,3), size(out,4))
    out
end

"""
```
|
E4a --
|
E4b --
|
```
"""
@∇ function contract_E4(E4a, E4b)
    @tensor out[m1,m2,m3,m4] := E4a[m1,p1, m3]*E4b[p1, m2, m4]
    out = reshape(out, size(out,1), size(out,2), :)
    out
end

"""
```
   |      |
-- T1 -h- T2 --
   |      |
```
"""
@∇ function contract_hor_Th(h, T1, T2)
    @tensor out[m1,m2,m3,m4,m5,m6] := T1[i1,j1, m1,m3,m4,p1]*T2[i2,j2, m2,p1,m5,m6]*h[i1,i2,j1,j2]
    dims = size(out)
    out = reshape(out, dims[1]*dims[2], dims[3], dims[4]*dims[5], dims[6])
    out
end

"""
```
   |      |
-- T1 --- T2 --
   |      |
```
"""
@∇ function contract_hor_T(T1, T2)
    @tensor out[m1,m2,m3,m4,m5,m6] := T1[m1,m3,m4,p1]*T2[m2,p1,m5,m6]
    dims = size(out)
    out = reshape(out, dims[1]*dims[2], dims[3], dims[4]*dims[5], dims[6])
    out
end

"""
```
   |        
-- T1 -- 
   h     
-- T2 --  
   |    
```
"""
@∇ function contract_ver_Th(h, T1, T2)
    @tensor out[m1,m2,m3,m4,m5,m6] := T1[i1,j1, m1,m2,p1,m5]*T2[i2,j2, p1,m3,m4,m6]*h[i1,i2,j1,j2]
    dims = size(out)
    out = reshape(out, dims[1], dims[2]*dims[3], dims[4], dims[5]*dims[6])
    out
end

"""
```
   |        
-- T1 -- 
   |     
-- T2 --  
   |    
```
"""
@∇ function contract_ver_T(T1, T2)
    @tensor out[m1,m2,m3,m4,m5,m6] := T1[m1,m2,p1,m5]*T2[p1,m3,m4,m6]
    dims = size(out)
    out = reshape(out, dims[1], dims[2]*dims[3], dims[4], dims[5]*dims[6])
    out
end

"""
    proj_left

return C1, E4, C4
```
C1 -- E1 --       
  \\  /
   Pl
   | 
   |
   Pld
  /  \\
E4 -- T --
  \\  /
   Pl
   |
   |
   Pld
  /  \\
C4 -- E3 --       
```
"""
@∇ function proj_left(Pl, Pld, C1, E1, E4, T, C4, E3)
    @tensor newC1[m1,m2,m3] := C1[m1, p1]*E1[p1,m2,m3]
    newC1 = reshape(newC1, :, size(newC1,3))
    @tensor newC1[m1,m2] := newC1[p1,m2]*Pl[p1,m1]

    @tensor newE4[m1,m2,m3,m4,m5] := E4[m1,m3,p1]*T[m2,p1,m4,m5]
    newE4 = reshape(newE4, size(newE4,1)*size(newE4,2), size(newE4,3)*size(newE4,4), :)
    @tensor newE4[m1,m2,m3] := Pld[m1,p1]*newE4[p1,p2,m3]*Pl[p2,m2]
    
    @tensor newC4[m1,m2,m3] := C4[m1, p1]*E3[m2,p1,m3]
    newC4 = reshape(newC4, :, size(newC4,3))
    @tensor newC4[m1,m2] := Pld[m1,p1]*newC4[p1,m2]

    newC1/maximum(abs, newC1), newE4/maximum(abs, newE4), newC4/maximum(abs, newC4)
end

"""
    proj_right

return C2, E2, C3
```
-- E1 -- C2       
     \\  /
      Pr
      | 
      |
      Prd
     /  \\
--- T -- E2 
     \\  /
      Pr
      |
      |
      Prd
     /  \\
-- E3 -- C3       
```
"""
@∇ function proj_right(Pr, Prd, C2, E1, E2, T, C3, E3)
    @tensor newC2[m1,m2,m3] := E1[m1,m3,p1]*C2[p1, m2] 
    newC2 = reshape(newC2, size(newC2,1), :)
    @tensor newC2[m1,m2] := newC2[m1,p1]*Pr[p1,m2]

    @tensor newE2[m1,m2,m3,m4,m5] :=T[m2,m3,m5,p1]*E2[m1,p1,m4]
    newE2 = reshape(newE2, size(newE2,1)*size(newE2,2), :, size(newE2,4)*size(newE2,5))
    @tensor newE2[m1,m2,m3] := Prd[m1,p1]*newE2[p1,m2,p2]*Pr[p2,m3]
    
    @tensor newC3[m1,m2,m3] := E3[m2,m3,p1]*C3[m1, p1]
    newC3 = reshape(newC3, :, size(newC3,3))
    @tensor newC3[m1,m2] := Prd[m1,p1]*newC3[p1,m2]

    newC2/maximum(abs,newC2), newE2/maximum(abs,newE2), newC3/maximum(abs,newC3)
end

"""
    proj_top

return C1, E1, C2
```
C1 \\             /  E1 \\            / C2 
|    - Pt - Ptd -   |   - Pt - Ptd -  |
E4 /             \\  T  /            \\ E2
|                   |                 |
```
"""
@∇ function proj_top(Pt, Ptd, C1, E4, E1, T, C2, E2) 
    @tensor newC1[m1,m2,m3] := C1[p1, m2]*E4[p1, m1, m3] 
    newC1 = reshape(newC1, size(newC1,1), :)
    @tensor newC1[m1,m2] := newC1[m1,p1]*Pt[p1,m2]

    @tensor newE1[m1,m2,m3,m4,m5] :=E1[m1,p1,m4]*T[p1,m2,m3,m5]
    newE1 = reshape(newE1, size(newE1,1)*size(newE1,2), :, size(newE1,4)*size(newE1,5))
    @tensor newE1[m1,m2,m3] := Ptd[m1,p1]*newE1[p1,m2,p2]*Pt[p2,m3]
    
    @tensor newC2[m1,m2,m3] := C2[m1,p1]*E2[p1,m2,m3]
    newC2 = reshape(newC2, :, size(newC2,3))
    @tensor newC2[m1,m2] := Ptd[m1,p1]*newC2[p1,m2]

    newC1/maximum(abs,newC1), newE1/maximum(abs,newE1), newC2/maximum(abs,newC2)
end

"""
    proj_bottom

return C4, E3, C3
```
|                   |                 |
E4 \\             /  T  \\            / E2 
|    - Pb - Pbd -   |   - Pb - Pbd -  |
C4 /             \\  E3 /            \\ C3
```
"""
@∇ function proj_bottom(Pb, Pbd, C4, E4, E3, T, C3, E2) 
    @tensor newC4[m1,m2,m3] := E4[m1,p1,m3]*C4[p1, m2] 
    newC4 = reshape(newC4, size(newC4,1), :)
    @tensor newC4[m1,m2] := newC4[m1,p1]*Pb[p1,m2]

    @tensor newE3[m1,m2,m3,m4,m5] := T[m1,m3,p1,m5]*E3[p1,m2,m4]
    newE3 = reshape(newE3, :, size(newE3,2)*size(newE3,3), size(newE3,4)*size(newE3,5))
    @tensor newE3[m1,m2,m3] := Pbd[m2,p1]*newE3[m1,p1,p2]*Pb[p2,m3]
    
    @tensor newC3[m1,m2,m3] := E2[m1,m3,p1]*C3[p1, m2]
    newC3 = reshape(newC3, size(newC3,1), :)
    @tensor newC3[m1,m2] := Pbd[m2,p1]*newC3[m1,p1]

    newC4/maximum(abs,newC4), newE3/maximum(abs,newE3), newC3/maximum(abs,newC3)
end

############ 

"""
    contract_ul_env(C1,E1,E4,T)

Return a matrix χD²*χD² 

contraction order:
```
C1 --1-- E1 -- -3
|        |
2        3
|        |
E4 --4-- T --- -4 
|        |
-1       -2
```
"""
@∇ function contract_ul_env(C1,E1,E4,T)
    @tensor UL[m1,m2,m3,m4] := C1[p2,p1]*E1[p1,p3,m3]*E4[p2,m1,p4]*T[p3,p4,m2,m4]
    UL = reshape(UL, size(UL,1)*size(UL,2), :)
    UL
end

"""
    contract_ur_env(C2,E1,E2,T)

Return a matrix χD²*χD² 

contraction order:
```
-1 -- E1 --1-- C2
      |        |
      3        2
      |        |
-2 -- T ---4-- E2 
      |        |
     -4       -3
```
"""
@∇ function contract_ur_env(C2,E1,E2,T)
    @tensor UR[m1,m2,m3,m4] := C2[p1,p2]*E1[m1,p3,p1]*E2[p2,p4,m3]*T[p3,m2,m4,p4]
    UR = reshape(UR, size(UR,1)*size(UR,2), :)
    UR
end

"""
    contract_bl_env(C4,E3,E4,T)

Return a matrix χD²*χD² 

contraction order:
```
-1       -2
|        |
E4 --3-- T --- -4 
|        |
2        4
|        |
C4 --1-- E3 -- -3
```
"""
@∇ function contract_bl_env(C4,E3,E4,T)
    @tensor BL[m1,m2,m3,m4] := C4[p2,p1]*E3[p4,p1,m3]*E4[m1,p2,p3]*T[m2,p3,p4,m4]
    BL = reshape(BL, size(BL,1)*size(BL,2), :)
    BL
end

"""
    contract_br_env(C3,E3,E2,T)

Return a matrix χD²*χD² 

contraction order:
```
     -2       -1
      |        |
-4 -- T ---4-- E2 
      |        |
      3        2
      |        |
-3 -- E3 --1-- C3
```
"""
@∇ function contract_br_env(C3,E3,E2,T)
    @tensor BR[m1,m2,m3,m4] := C3[p2,p1]*E3[p3,m3,p1]*E2[m1,p4,p2]*T[m2,m4,p3,p4]
    BR = reshape(BR, size(BR,1)*size(BR,2), :)
    BR
end

########## 
"""
    contract_env

```
C1 - 0 - E1 - 3 - C2
|        |        |     
0        0        0
|        |        |
E4 - 0 - AB - 4 - E2  = N
|        |        |
1        2        5
|        |        |
C4 - 0 - E3 - 6 - C3
```
"""
@∇ function contract_env(C1, C2, C3, C4, E1, E2, E3, E4, A, B)
    T = transfer_matrix(A, B)
    contract_env(C1, C2, C3, C4, E1, E2, E3, E4, T)
end

@∇ function contract_env(C1, C2, C3, C4, E1, E2, E3, E4, T)
    ul = contract_ul_env(C1, E1, E4, T)

    @tensor C4E3[m1,m2,m3] := C4[m1, p1]*E3[m2, p1, m3]
    @tensor C2E2[m1,m2,m3] := C2[m1, p1]*E2[p1, m2, m3]
    C4E3 = reshape(C4E3, :, size(C4E3, 3))
    C2E2 = reshape(C2E2, :, size(C2E2, 3))

    @tensor out[m1,m2] :=  ul[p1,p2]*C4E3[p1,m1]*C2E2[p2,m2]

    out = tr(out*C3)
    out
end

"""
    contract_env_dA

Return a tensor(dD⁴)    

```
C1 - 1 - E1 - 7 - C2
|        ||       |     
2        3        0
|        |        |        |
E4 = 4 - A  - 8 = E2  = -- dA -- 
|        |        |        | \\
5        6        9
|        ||       |
C4 - 0 - E3 - 10- C3
```
"""
@∇ function contract_env_dA(C1, C2, C3, C4, E1, E2, E3, E4, A)
    D = size(A, 2)
    @tensor out[m1,m2,m3,m4] := C1[p1,p2]*E1[p2,m3,m4]*E4[p1,m1,m2]
    dim = size(out)
    out = reshape(out, dim[1], D,D, D,D, dim[4])
    @tensor out[m0,m1,m2, m3,m4,m5,m6] := out[m3, p2,m2, p1,m1, m5]*A[m0, p1,p2,m4,m6]

    @tensor C4E3[m1,m2,m3] := C4[m1, p1]*E3[m2, p1, m3]
    C4E3 = reshape(C4E3, size(C4E3, 1), D,D, size(C4E3, 3))
    @tensor C2E2[m1,m2,m3] := C2[m1, p1]*E2[p1, m2, m3]
    C2E2 = reshape(C2E2, size(C2E2, 1), D,D, size(C2E2, 3))

    @tensor out[m0,m1,m2,m3,m4, m5,m6] := out[m0,m1,m2, p1, p2, p3, p4]*C4E3[p1, p2,m3, m5]*C2E2[p3, p4,m4, m6]
    @tensor out[m0,m1,m2,m3,m4] := out[m0,m1,m2,m3,m4, p1,p2]*C3[p2,p1]

    out
end

##### exc env contraction
"""
    proj_left_B

return C1_B, E4_B, C4_B
```
(C1_B -- E1 --  +  C1 -- E1_B -- )⋅exp(-ikₓ)
            \\  /
            Pl
            | 
            |
            Pld
            /  \\
(E4_B -- T --  +  E4 -- T_B -- )⋅exp(-ikₓ)
            \\  /
            Pl
            |
            |
            Pld
            /  \\
(C4_B -- E3 --  +  C4 -- E3_B -- )⋅exp(-ikₓ)
```
"""
@∇ function proj_left_B(kx, Pl, Pld, C1_B, E1, C1, E1_B, 
                                  E4_B, T, E4, T_B, 
                                  C4_B, E3, C4, E3_B)

    @tensor newC1_B[m1,m2,m3] := (C1_B[m1, p1]*E1[p1,m2,m3] + C1[m1, p1]*E1_B[p1,m2,m3])*exp(-im*kx)
    newC1_B = reshape(newC1_B, :, size(newC1_B,3))
    @tensor newC1_B[m1,m2] := newC1_B[p1,m2]*Pl[p1,m1]

    @tensor newE4_B[m1,m2,m3,m4,m5] := (E4_B[m1,m3,p1]*T[m2,p1,m4,m5] + E4[m1,m3,p1]*T_B[m2,p1,m4,m5])*exp(-im*kx)
    newE4_B = reshape(newE4_B, size(newE4_B,1)*size(newE4_B,2), size(newE4_B,3)*size(newE4_B,4), :)
    @tensor newE4_B[m1,m2,m3] := Pld[m1,p1]*newE4_B[p1,p2,m3]*Pl[p2,m2]
    
    @tensor newC4_B[m1,m2,m3] := (C4_B[m1, p1]*E3[m2,p1,m3] + C4[m1, p1]*E3_B[m2,p1,m3])*exp(-im*kx)
    newC4_B = reshape(newC4_B, :, size(newC4_B,3))
    @tensor newC4_B[m1,m2] := Pld[m1,p1]*newC4_B[p1,m2]

    newC1_B/maximum(abs,newC1_B), newE4_B/maximum(abs,newE4_B), newC4_B/maximum(abs,newC4_B)
end

"""
    proj_left_Bd

return C1_Bd, E4_Bd, C4_Bd
```
(C1_Bd -- E1 --  +  C1 -- E1_Bd -- )⋅exp(ikₓ)
                \\  /
                Pl
                | 
                |
                Pld
                /  \\
(E4_Bd -- T --  +  E4 -- T_Bd -- )⋅exp(ikₓ)
                \\  /
                Pl
                |
                |
                Pld
                /  \\
(C4_Bd -- E3 --  +  C4 -- E3_Bd -- )⋅exp(ikₓ)
```
"""
function proj_left_Bd(kx, Pl, Pld, C1_Bd, E1, C1, E1_Bd, 
                                   E4_Bd, T, E4, T_Bd, 
                                   C4_Bd, E3, C4, E3_Bd)

    proj_left_B(-kx, Pl, Pld, C1_Bd, E1, C1, E1_Bd, 
                              E4_Bd, T, E4, T_Bd, 
                              C4_Bd, E3, C4, E3_Bd)
end

"""
    proj_left_BB

return C1_BB, E4_BB, C4_BB
```
(C1_BB -- E1 --  +  C1 -- E1_BB  +  C1_B -- E1_Bd --  +  C1_Bd -- E1_B -- )
                                \\  /
                                Pl
                                | 
                                |
                                Pld
                                /  \\
(E4_BB -- T --  +  E4 -- T_BB --  +  E4_B -- T_Bd --  +  E4_Bd -- T_B -- )
                                \\  /
                                Pl
                                |
                                |
                                Pld
                                /  \\
(C4_BB -- E3 --  +  C4 -- E3_BB  +  C4_B -- E3_Bd --  +  C4_Bd -- E3_B -- )
```
"""
@∇ function proj_left_BB(Pl, Pld, C1_BB,E1, C1,E1_BB, C1_B,E1_Bd, C1_Bd,E1_B, 
                               E4_BB,T, E4,T_BB, E4_B,T_Bd, E4_Bd,T_B, 
                               C4_BB,E3, C4,E3_BB, C4_B,E3_Bd, C4_Bd,E3_B)
    @tensor newC1_BB[m1,m2,m3] := C1_BB[m1, p1]*E1[p1,m2,m3] + C1[m1, p1]*E1_BB[p1,m2,m3] + C1_B[m1, p1]*E1_Bd[p1,m2,m3] + C1_Bd[m1, p1]*E1_B[p1,m2,m3]
    newC1_BB = reshape(newC1_BB, :, size(newC1_BB,3))
    @tensor newC1_BB[m1,m2] := newC1_BB[p1,m2]*Pl[p1,m1]

    @tensor newE4_BB[m1,m2,m3,m4,m5] := E4_BB[m1,m3,p1]*T[m2,p1,m4,m5] + E4[m1,m3,p1]*T_BB[m2,p1,m4,m5] + E4_B[m1,m3,p1]*T_Bd[m2,p1,m4,m5] + E4_Bd[m1,m3,p1]*T_B[m2,p1,m4,m5]
    newE4_BB = reshape(newE4_BB, size(newE4_BB,1)*size(newE4_BB,2), size(newE4_BB,3)*size(newE4_BB,4), :)
    @tensor newE4_BB[m1,m2,m3] := Pld[m1,p1]*newE4_BB[p1,p2,m3]*Pl[p2,m2]
    
    @tensor newC4_BB[m1,m2,m3] := C4_BB[m1, p1]*E3[m2,p1,m3] + C4[m1, p1]*E3_BB[m2,p1,m3] + C4_B[m1, p1]*E3_Bd[m2,p1,m3] + C4_Bd[m1, p1]*E3_B[m2,p1,m3]
    newC4_BB = reshape(newC4_BB, :, size(newC4_BB,3))
    @tensor newC4_BB[m1,m2] := Pld[m1,p1]*newC4_BB[p1,m2]

    newC1_BB/maximum(abs,newC1_BB), newE4_BB/maximum(abs,newE4_BB), newC4_BB/maximum(abs,newC4_BB)
end

"""
    proj_right_B

return C2_B, E2_B, C3_B
```
(-- E1 -- C2_B  +  -- E1_B -- C2 )⋅exp(ikₓ)
            \\  /
            Pr
            | 
            |
            Prd
            /  \\
( -- T -- E2_B  +  -- T_B -- E2 )⋅exp(ikₓ)
            \\  /
            Pr
            |
            |
            Prd
            /  \\
(-- E3 -- C3_B  +  -- E3_B -- C3)⋅exp(ikₓ)
```
"""
@∇ function proj_right_B(kx, Pr, Prd, C2_B, E1, C2, E1_B, 
                                   E2_B, T, E2, T_B, 
                                   C3_B, E3, C3, E3_B)
    
    @tensor newC2_B[m1,m2,m3] := (E1[m1,m3,p1]*C2_B[p1, m2] + E1_B[m1,m3,p1]*C2[p1, m2])*exp(im*kx)
    newC2_B = reshape(newC2_B, size(newC2_B,1), :)
    @tensor newC2_B[m1,m2] := newC2_B[m1,p1]*Pr[p1,m2]

    @tensor newE2_B[m1,m2,m3,m4,m5] := (T[m2,m3,m5,p1]*E2_B[m1,p1,m4] + T_B[m2,m3,m5,p1]*E2[m1,p1,m4])*exp(im*kx)
    newE2_B = reshape(newE2_B, size(newE2_B,1)*size(newE2_B,2), :, size(newE2_B,4)*size(newE2_B,5))
    @tensor newE2_B[m1,m2,m3] := Prd[m1,p1]*newE2_B[p1,m2,p2]*Pr[p2,m3]
    
    @tensor newC3_B[m1,m2,m3] := (E3[m2,m3,p1]*C3_B[m1, p1] + E3_B[m2,m3,p1]*C3[m1, p1])*exp(im*kx)
    newC3_B = reshape(newC3_B, :, size(newC3_B,3))
    @tensor newC3_B[m1,m2] := Prd[m1,p1]*newC3_B[p1,m2]

    newC2_B/maximum(abs,newC2_B), newE2_B/maximum(abs,newE2_B), newC3_B/maximum(abs,newC3_B)
end

"""
    proj_left_Bd

return C2_Bd, E2_Bd, C3_Bd
```
(-- E1 -- C2_Bd  +  -- E1_Bd -- C2 )⋅exp(-ikₓ)
            \\  /
            Pr
            | 
            |
            Prd
            /  \\
( -- T -- E2_Bd  +  -- T_Bd -- E2 )⋅exp(-ikₓ)
            \\  /
            Pr
            |
            |
            Prd
            /  \\
(-- E3 -- C3_Bd  +  -- E3_Bd -- C3)⋅exp(-ikₓ)
```
"""
function proj_right_Bd(kx, Pr, Prd, C2_Bd, E1, C2, E1_Bd, 
                                    E2_Bd, T,  E2, T_Bd, 
                                    C3_Bd, E3, C3, E3_Bd)

    proj_right_B(-kx, Pr, Prd, C2_Bd, E1, C2, E1_Bd, 
                               E2_Bd, T,  E2, T_Bd, 
                               C3_Bd, E3, C3, E3_Bd)
end

"""
    proj_right_BB

return C2_BB, E2_BB, C3_BB
```
(-- E1 -- C2_BB  +  -- E1_BB -- C2  +  -- E1_Bd -- C2_B  +  -- E1_B -- C2_Bd)
                                \\  /
                                Pr
                                | 
                                |
                                Prd
                                /  \\
(-- T -- E2_BB +  -- T_BB -- E2  +  -- T_Bd -- E2_B  +  -- T_B -- E2_Bd)
                                \\  /
                                Pr
                                |
                                |
                                Prd
                                /  \\
(-- E3 -- C3_BB  +  -- E3_BB -- C3 +  -- E3_Bd -- C3_B  +  -- E3_B -- C3_Bd)
```
"""
@∇ function proj_right_BB(Pr, Prd, C2_BB,E1, C2,E1_BB, C2_B,E1_Bd, C2_Bd,E1_B, 
                                E2_BB,T,  E2,T_BB,  E2_B,T_Bd,  E2_Bd,T_B, 
                                C3_BB,E3, C3,E3_BB, C3_B,E3_Bd, C3_Bd,E3_B)
    
    @tensor newC2_BB[m1,m2,m3] := E1[m1,m3,p1]*C2_BB[p1, m2] + E1_BB[m1,m3,p1]*C2[p1, m2] + E1_Bd[m1,m3,p1]*C2_B[p1, m2] + E1_B[m1,m3,p1]*C2_Bd[p1, m2]
    newC2_BB = reshape(newC2_BB, size(newC2_BB,1), :)
    @tensor newC2_BB[m1,m2] := newC2_BB[m1,p1]*Pr[p1,m2]

    @tensor newE2_BB[m1,m2,m3,m4,m5] := T[m2,m3,m5,p1]*E2_BB[m1,p1,m4] + T_BB[m2,m3,m5,p1]*E2[m1,p1,m4] + T_Bd[m2,m3,m5,p1]*E2_B[m1,p1,m4] + T_B[m2,m3,m5,p1]*E2_Bd[m1,p1,m4]
    newE2_BB = reshape(newE2_BB, size(newE2_BB,1)*size(newE2_BB,2), :, size(newE2_BB,4)*size(newE2_BB,5))
    @tensor newE2_BB[m1,m2,m3] := Prd[m1,p1]*newE2_BB[p1,m2,p2]*Pr[p2,m3]
    
    @tensor newC3_BB[m1,m2,m3] := E3[m2,m3,p1]*C3_BB[m1, p1] + E3_BB[m2,m3,p1]*C3[m1, p1] + E3_Bd[m2,m3,p1]*C3_B[m1, p1] + E3_B[m2,m3,p1]*C3_Bd[m1, p1]
    newC3_BB = reshape(newC3_BB, :, size(newC3_BB,3))
    @tensor newC3_BB[m1,m2] := Prd[m1,p1]*newC3_BB[p1,m2]

    newC2_BB/maximum(abs,newC2_BB), newE2_BB/maximum(abs,newE2_BB), newC3_BB/maximum(abs,newC3_BB)
end

"""
    proj_top_B

return C1_B, E1_B, C2_B
```
exp(-iky)⋅[
C1_B                     E1_B                     C2_B 
|                        |                        |
E4                       T                        E2
|    \\              /    |    \\              /    |
+      - Pt - Ptd -      +      - Pt - Ptd -      +
C1   /              \\    E1   /              \\    C2
|                        |                        |
E4_B                     T_B                      E2_B
|                        |                        |
]
```
"""
@∇ function proj_top_B(ky, Pt, Ptd, C1_B,E4, C1,E4_B,
                                 E1_B,T,  E1,T_B,
                                 C2_B,E2, C2,E2_B)  

    @tensor newC1_B[m1,m2,m3] := (C1_B[p1, m2]*E4[p1, m1, m3] + C1[p1, m2]*E4_B[p1, m1, m3])*exp(-im*ky) 
    newC1_B = reshape(newC1_B, size(newC1_B,1), :)
    @tensor newC1_B[m1,m2] := newC1_B[m1,p1]*Pt[p1,m2]

    @tensor newE1_B[m1,m2,m3,m4,m5] := (E1_B[m1,p1,m4]*T[p1,m2,m3,m5] + E1[m1,p1,m4]*T_B[p1,m2,m3,m5])*exp(-im*ky)
    newE1_B = reshape(newE1_B, size(newE1_B,1)*size(newE1_B,2), :, size(newE1_B,4)*size(newE1_B,5))
    @tensor newE1_B[m1,m2,m3] := Ptd[m1,p1]*newE1_B[p1,m2,p2]*Pt[p2,m3]
    
    @tensor newC2_B[m1,m2,m3] := (C2_B[m1,p1]*E2[p1,m2,m3] + C2[m1,p1]*E2_B[p1,m2,m3])*exp(-im*ky)
    newC2_B = reshape(newC2_B, :, size(newC2_B,3))
    @tensor newC2_B[m1,m2] := Ptd[m1,p1]*newC2_B[p1,m2]

    newC1_B/maximum(abs,newC1_B), newE1_B/maximum(abs,newE1_B), newC2_B/maximum(abs,newC2_B)
end

"""
    proj_top_Bd

return C1_Bd, E1_Bd, C2_Bd
```
exp(iky)⋅[
C1_Bd                    E1_Bd                    C2_Bd 
|                        |                        |
E4                       T                        E2
|    \\              /    |    \\              /    |
+      - Pt - Ptd -      +      - Pt - Ptd -      +
C1   /              \\    E1   /              \\    C2
|                        |                        |
E4_Bd                    T_Bd                     E2_Bd
|                        |                        |
]
```
"""
function proj_top_Bd(ky, Pt, Ptd, C1_Bd,E4, C1,E4_Bd,
                                  E1_Bd,T,  E1,T_Bd,
                                  C2_Bd,E2, C2,E2_Bd)    

    proj_top_B(-ky, Pt, Ptd, C1_Bd,E4, C1,E4_Bd,
                             E1_Bd,T,  E1,T_Bd,
                             C2_Bd,E2, C2,E2_Bd) 
end

"""
    proj_top_BB

return C1_BB, E1_BB, C2_BB
```
C1_BB                    E1_BB                    C2_BB 
|                        |                        |
E4                       T                        E2
|    \\              /    |    \\              /    |
+      - Pt - Ptd -      +      - Pt - Ptd -      +
C1   /              \\    E1   /              \\    C2
|                        |                        |
E4_BB                    T_BB                     E2_BB
|                        |                        |
+
C1_B                     E1_B                     C2_B 
|                        |                        |
E4_Bd                    T_Bd                     E2_Bd
|     \\              /   |     \\              /   |
+      - Pt - Ptd -      +       - Pt - Ptd -     +
C1_Bd /              \\   E1_Bd /              \\   C2_Bd
|                        |                        |
E4_B                     T_B                      E2_B
|                        |                        |
```
"""
@∇ function proj_top_BB(Pt, Ptd, C1_BB,E4, C1,E4_BB, C1_B,E4_Bd, C1_Bd,E4_B,
                              E1_BB,T,  E1,T_BB,  E1_B,T_Bd,  E1_Bd, T_B,
                              C2_BB,E2, C2,E2_BB, C2_B,E2_Bd, C2_Bd, E2_B)
                              
    @tensor newC1_BB[m1,m2,m3] := C1_BB[p1, m2]*E4[p1, m1, m3] + C1[p1, m2]*E4_BB[p1, m1, m3] + C1_B[p1, m2]*E4_Bd[p1, m1, m3] + C1_Bd[p1, m2]*E4_B[p1, m1, m3] 
    newC1_BB = reshape(newC1_BB, size(newC1_BB,1), :)
    @tensor newC1_BB[m1,m2] := newC1_BB[m1,p1]*Pt[p1,m2]

    @tensor newE1_BB[m1,m2,m3,m4,m5] := E1_BB[m1,p1,m4]*T[p1,m2,m3,m5] + E1[m1,p1,m4]*T_BB[p1,m2,m3,m5] + E1_B[m1,p1,m4]*T_Bd[p1,m2,m3,m5] + E1_Bd[m1,p1,m4]*T_B[p1,m2,m3,m5]
    newE1_BB = reshape(newE1_BB, size(newE1_BB,1)*size(newE1_BB,2), :, size(newE1_BB,4)*size(newE1_BB,5))
    @tensor newE1_BB[m1,m2,m3] := Ptd[m1,p1]*newE1_BB[p1,m2,p2]*Pt[p2,m3]
    
    @tensor newC2_BB[m1,m2,m3] := C2_BB[m1,p1]*E2[p1,m2,m3] + C2[m1,p1]*E2_BB[p1,m2,m3] + C2_B[m1,p1]*E2_Bd[p1,m2,m3] + C2_Bd[m1,p1]*E2_B[p1,m2,m3]
    newC2_BB = reshape(newC2_BB, :, size(newC2_BB,3))
    @tensor newC2_BB[m1,m2] := Ptd[m1,p1]*newC2_BB[p1,m2]

    newC1_BB/maximum(abs,newC1_BB), newE1_BB/maximum(abs,newE1_BB), newC2_BB/maximum(abs,newC2_BB)
end


"""
    proj_bottom_B

return C4_B, E3_B, C3_B
```
[
|                        |                        |
E4                       T                        E2
|                        |                        |
C4_B \\              /    E3_B \\              /    C3_B 
+      - Pb - Pbd -      +      - Pb - Pbd -      +
|    /              \\    |    /              \\    |
E4_B                     T_B                      E2_B
|                        |                        |
C4                       E3                       C3
]⋅exp(iky)
```
"""
@∇ function proj_bottom_B(ky, Pb, Pbd, C4_B,E4, C4,E4_B,
                                    E3_B,T, E3,T_B,
                                    C3_B,E2, C3,E2_B)

    @tensor newC4_B[m1,m2,m3] := (E4[m1,p1,m3]*C4_B[p1, m2] + E4_B[m1,p1,m3]*C4[p1, m2])*exp(im*ky)
    newC4_B = reshape(newC4_B, size(newC4_B,1), :)
    @tensor newC4_B[m1,m2] := newC4_B[m1,p1]*Pb[p1,m2]

    @tensor newE3_B[m1,m2,m3,m4,m5] := (T[m1,m3,p1,m5]*E3_B[p1,m2,m4] + T_B[m1,m3,p1,m5]*E3[p1,m2,m4])*exp(im*ky)
    newE3_B = reshape(newE3_B, :, size(newE3_B,2)*size(newE3_B,3), size(newE3_B,4)*size(newE3_B,5))
    @tensor newE3_B[m1,m2,m3] := Pbd[m2,p1]*newE3_B[m1,p1,p2]*Pb[p2,m3]
    
    @tensor newC3_B[m1,m2,m3] := (E2[m1,m3,p1]*C3_B[p1, m2] + E2_B[m1,m3,p1]*C3[p1, m2])*exp(im*ky)
    newC3_B = reshape(newC3_B, size(newC3_B,1), :)
    @tensor newC3_B[m1,m2] := Pbd[m2,p1]*newC3_B[m1,p1]

    newC4_B/maximum(abs,newC4_B), newE3_B/maximum(abs,newE3_B), newC3_B/maximum(abs,newC3_B)
end

"""
    proj_bottom_Bd

return C4_Bd, E3_Bd, C3_Bd
```
[
|                        |                        |
E4                       T                        E2
|                        |                        |
C4_Bd \\              /   E3_Bd \\              /   C3_Bd 
+      - Pb - Pbd -      +      - Pb - Pbd -      +
|     /              \\   |     /              \\   |
E4_Bd                    T_Bd                     E2_Bd
|                        |                        |
C4                       E3                       C3
]⋅exp(-iky)
```
"""
function proj_bottom_Bd(ky, Pb, Pbd, C4_Bd,E4, C4,E4_Bd,
                                     E3_Bd,T,  E3,T_Bd,
                                     C3_Bd,E2, C3,E2_Bd) 

    proj_bottom_B(-ky, Pb, Pbd, C4_Bd,E4, C4,E4_Bd,
                                E3_Bd,T,  E3,T_Bd,
                                C3_Bd,E2, C3,E2_Bd) 
end

"""
    proj_bottom_BB

return C4_BB, E3_BB, C3_BB
```
|                        |                        |
E4                       T                        E2
|                        |                        |
C4_BB \\              /   E3_BB \\              /   C3_BB 
+      - Pb - Pbd -      +      - Pb - Pbd -      +
|     /              \\   |     /              \\   |
E4_BB                    T_BB                     E2_BB
|                        |                        |
C4                       E3                       C3
+
|                        |                        |
E4_Bd                    T_Bd                     E2_Bd
|                        |                        |
C4_B \\              /    E3_B \\              /    C3_B 
+      - Pb - Pbd -      +      - Pb - Pbd -      +
|    /              \\    |    /              \\    |
E4_B                     T_B                      E2_B
|                        |                        |
C4_Bd                    E3_Bd                    C3_Bd
```
"""
@∇ function proj_bottom_BB(Pb, Pbd, C4_BB,E4, C4,E4_BB, C4_B,E4_Bd, C4_Bd, E4_B,
                                 E3_BB,T,  E3,T_BB,  E3_B,T_Bd,  E3_Bd, T_B,
                                 C3_BB,E2, C3,E2_BB, C3_B,E2_Bd, C3_Bd, E2_B)    

    @tensor newC4_BB[m1,m2,m3] := E4[m1,p1,m3]*C4_BB[p1, m2] + E4_BB[m1,p1,m3]*C4[p1, m2] + E4_Bd[m1,p1,m3]*C4_B[p1, m2] + E4_B[m1,p1,m3]*C4_Bd[p1, m2]
    newC4_BB = reshape(newC4_BB, size(newC4_BB,1), :)
    @tensor newC4_BB[m1,m2] := newC4_BB[m1,p1]*Pb[p1,m2]

    @tensor newE3_BB[m1,m2,m3,m4,m5] := T[m1,m3,p1,m5]*E3_BB[p1,m2,m4] + T_BB[m1,m3,p1,m5]*E3[p1,m2,m4] + T_Bd[m1,m3,p1,m5]*E3_B[p1,m2,m4] + T_B[m1,m3,p1,m5]*E3_Bd[p1,m2,m4]
    newE3_BB = reshape(newE3_BB, :, size(newE3_BB,2)*size(newE3_BB,3), size(newE3_BB,4)*size(newE3_BB,5))
    @tensor newE3_BB[m1,m2,m3] := Pbd[m2,p1]*newE3_BB[m1,p1,p2]*Pb[p2,m3]
    
    @tensor newC3_BB[m1,m2,m3] := E2[m1,m3,p1]*C3_BB[p1, m2] + E2_BB[m1,m3,p1]*C3[p1, m2] + E2_Bd[m1,m3,p1]*C3_B[p1, m2] + E2_B[m1,m3,p1]*C3_Bd[p1, m2]
    newC3_BB = reshape(newC3_BB, size(newC3_BB,1), :)
    @tensor newC3_BB[m1,m2] := Pbd[m2,p1]*newC3_BB[m1,p1]

    newC4_BB/maximum(abs,newC4_BB), newE3_BB/maximum(abs,newE3_BB), newC3_BB/maximum(abs,newC3_BB)
end