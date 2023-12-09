function heisenberg(Jz=1)
    H = Jz * tout(Sz, Sz) - tout(Sx, Sx) - tout(Sy, Sy)

    [H, H]
end