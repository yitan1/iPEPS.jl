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
    proj_left

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
function proj_left(Pl, Pld, C1, E1, E4, T, C4, E3)
    @tensor newC1[m1,m2,m3] := C1[m1, p1]*E1[p1,m2,m3]
    newC1 = reshape(newC1, :, size(newC1,3))
    @tensor newC1[m1,m2] := newC1[p1,m2]*Pl[p1,m1]

    @tensor newE4[m1,m2,m3,m4,m5] := E4[m1,m3,p1]*T[m2,p1,m4,m5]
    newE4 = reshape(newE4, size(newE4,1)*size(newE4,2), size(newE4,3)*size(newE4,4), :)
    @tensor newE4[m1,m2,m3] := Pld[m1,p1]*newE4[p1,p2,m3]*Pl[p2,m2]
    
    @tensor newC4[m1,m2,m3] := C4[m1, p1]*E3[m2,p1,m3]
    newC4 = reshape(newC4, :, size(newC4,3))
    @tensor newC4[m1,m2] := Pld[m1,p1]*newC4[p1,m2]

    newC1/norm(newC1), newE4/norm(newE4), newC4/norm(newC4)
end

"""
    proj_right

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
function proj_right(Pr, Prd, C2, E1, E2, T, C3, E3)
    @tensor newC2[m1,m2,m3] := E1[m1,m3,p1]*C2[p1, m2] 
    newC2 = reshape(newC2, size(newC2,1), :)
    @tensor newC2[m1,m2] := newC2[m1,p1]*Pr[p1,m2]

    @tensor newE2[m1,m2,m3,m4,m5] :=T[m2,m3,m5,p1]*E2[m1,p1,m4]
    newE2 = reshape(newE2, size(newE2,1)*size(newE2,2), :, size(newE2,4)*size(newE2,5))
    @tensor newE2[m1,m2,m3] := Prd[m1,p1]*newE2[p1,m2,p2]*Pr[p2,m3]
    
    @tensor newC3[m1,m2,m3] := E3[m2,m3,p1]*C3[m1, p1]
    newC3 = reshape(newC3, :, size(newC3,3))
    @tensor newC3[m1,m2] := Prd[m1,p1]*newC3[p1,m2]

    newC2/norm(newC2), newE2/norm(newE2), newC3/norm(newC3)
end

"""
    proj_top

```
C1 \\             /  E1 \\            / C2 
|    - Pt - Ptd -   |   - Pt - Ptd -  |
E4 /             \\  T  /            \\ E2
|                   |                 |
```
"""
function proj_top(Pt, Ptd, C1, E4, E1, T, C2, E2) 
    @tensor newC1[m1,m2,m3] := C1[p1, m2]*E4[p1, m1, m3] 
    newC1 = reshape(newC1, size(newC1,1), :)
    @tensor newC1[m1,m2] := newC1[m1,p1]*Pt[p1,m2]

    @tensor newE1[m1,m2,m3,m4,m5] :=E1[m1,p1,m4]*T[p1,m2,m3,m5]
    newE1 = reshape(newE1, size(newE1,1)*size(newE1,2), :, size(newE1,4)*size(newE1,5))
    @tensor newE1[m1,m2,m3] := Ptd[m1,p1]*newE1[p1,m2,p2]*Pt[p2,m3]
    
    @tensor newC2[m1,m2,m3] := C2[m1,p1]*E2[p1,m2,m3]
    newC2 = reshape(newC2, :, size(newC2,3))
    @tensor newC2[m1,m2] := Ptd[m1,p1]*newC2[p1,m2]

    newC1/norm(newC1), newE1/norm(newE1), newC2/norm(newC2)
end

"""
    proj_bottom

```
|                   |                 |
E4 \\             /  T  \\            / E2 
|    - Pb - Pbd -   |   - Pb - Pbd -  |
C4 /             \\  E3 /            \\ C3
```
"""
function proj_bottom(Pb, Pbd, C4, E4, E3, T, C3, E2) 
    @tensor newC4[m1,m2,m3] := E4[m1,p1,m3]*C4[p1, m2] 
    newC4 = reshape(newC4, size(newC4,1), :)
    @tensor newC4[m1,m2] := newC4[m1,p1]*Pb[p1,m2]

    @tensor newE3[m1,m2,m3,m4,m5] := T[m1,m3,p1,m5]*E3[p1,m2,m4]
    newE3 = reshape(newE3, :, size(newE3,2)*size(newE3,3), size(newE3,4)*size(newE3,5))
    @tensor newE3[m1,m2,m3] := Pbd[m2,p1]*newE3[m1,p1,p2]*Pb[p2,m3]
    
    @tensor newC3[m1,m2,m3] := E2[m1,m3,p1]*C3[p1, m2]
    newC3 = reshape(newC3, size(newC3,1), :)
    @tensor newC3[m1,m2] := Pbd[m2,p1]*newC3[m1,p1]

    newC4/norm(newC4), newE3/norm(newE3), newC3/norm(newC3)
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
function contract_ur_env(C2,E1,E2,T)
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
function contract_bl_env(C4,E3,E4,T)
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
function contract_br_env(C3,E3,E2,T)
    @tensor BR[m1,m2,m3,m4] := C3[p2,p1]*E3[p3,m3,p1]*E2[m1,p4,p2]*T[m2,m4,p3,p4]
    BR = reshape(BR, size(BR,1)*size(BR,2), :)
    BR
end

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
function contract_env(C1, C2, C3, C4, E1, E2, E3, E4, A, B)
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
function contract_env_dA(C1, C2, C3, C4, E1, E2, E3, E4, A)
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