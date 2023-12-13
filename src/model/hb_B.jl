function hb_xx_k(Jx=1, Jy=1; Jz = 1, K = 0.1)
    hv = Jy * tout(tout(SI, sigmay), tout(sigmay, SI)) .+ (tout(tout(sigmax, sigmax), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmax, sigmax))) / 2 / 2 + K * tout(tout(sigmax, sigmaz), tout(sigmay, SI)) + K * tout(tout(SI, sigmay), tout(sigmaz, sigmax)) .|> real
    hh = Jz * tout(tout(SI, sigmaz), tout(sigmaz, SI)) .+ (tout(tout(sigmax, sigmax), tout(SI, SI)) .+ tout(tout(SI, SI), tout(sigmax, sigmax))) / 2 / 2 + K * tout(tout(sigmax, sigmay), tout(sigmaz, SI)) + K * tout(tout(SI, sigmaz), tout(sigmay, sigmax)).|> real
    
    [-hh, -hv]
end  