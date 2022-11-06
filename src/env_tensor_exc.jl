"""
    ExcEnvTensor

corner: Ci, B_Ci, Bd_Ci, BB_Ci;  
edge: Ei, B_Ei, Bd_Ei, BB_Ei;
    (where i = 1, 2, 3, 4)
"""
struct ExcEnvTensor 
    data::Vector{EnvTensor}
end