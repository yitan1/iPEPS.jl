function get_local_h(h)
    dims = size(h)
    d = sqrt(dims[1]) |> Int
    sl = []
    
    h = reshape(h, dims[1], :)
    u,s,v = wrap_svd(h)  
    s1 = u * diagm(sqrt.(s))
    sdots = diagm(sqrt.(s)) * v'
    
    s1 = reshape(s1, d, d, :)
    push!(sl, s1)
    
    for i in eachindex(dims[2:end-1])
        dl = size(sdots, 1)
        sdots = reshape(sdots, dl*dims[i], :)
        u,s,v = wrap_svd(sdots)
        si = u * diagm(sqrt.(s))

        si = reshape(si, dl, d, d, :)
        push!(sl, si)
        
        sdots = diagm(sqrt.(s)) * v'
    end
    
    dl = size(sdots, 1)
    s_end = reshape(sdots, dl, d, d)
    push!(sl, s_end)
    
    return sl
end

# TODO
function get_local_h2(h)
    hh, hv = h

    dims = size(h)
end

