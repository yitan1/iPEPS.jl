
function rg_step(tensor::CTMTensors)
    tensor, s = left_rg(tensor)
    tensor, s = right_rg(tensor)
    tensor, s = top_rg(tensor)
    tensor, s = bottom_rg(tensor)

    tensor, s
end

function left_rg(tensor)

end