function tcon(xs, ind_xs)
    ind_num = [count(<(0), ind_xs[i])  for i in eachindex(ind_xs) ] |> sum 
    ind_y = -collect(1:ind_num)
    tcon(xs, ind_xs, ind_y)
end

tcon(xs, ind_xs, ind_y) = wrap_ncon(xs, ind_xs, ind_y)

function wrap_ncon(xs, ind_xs, ind_y)
    EinCode(ind_xs, ind_y)(xs...)
end