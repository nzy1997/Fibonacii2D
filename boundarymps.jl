using LinearAlgebra
using OMEinsum
# using CUDA

function get_lnZ(L::Int; β=0.44068679350977147, χ=16, device=:cpu)
    @assert iseven(L) "L must be even"

    # Create Boltzmann matrix
    B = sqrt([exp(β) exp(-β); exp(-β) exp(β)])
    B = device == :cuda ? CuArray(B) : B
    
    # Create tensors A2, A3, A4
    A2 = B * B
    A3 = ein"i,j,k->ijk"(B[:,1], B[:,1], B[:,1]) + ein"i,j,k->ijk"(B[:,2], B[:,2], B[:,2])
    A4 = ein"i,j,k,l->ijkl"(B[:,1], B[:,1], B[:,1], B[:,1]) + 
         ein"i,j,k,l->ijkl"(B[:,2], B[:,2], B[:,2], B[:,2])
    
    # Initialize tensors array
    tensors = []
    push!(tensors, [i == 1 ? reshape(A2, 1, 2, 2) : 
                    i == L ? reshape(A2, 2, 2, 1) : 
                    A3 for i in 1:L])
    for j in 2:(L÷2)
        push!(tensors, [i == 1 ? reshape(A3, 1, 2, 2, 2) :
                        i == L ? reshape(A3, 2, 2, 1, 2) :
                        A4 for i in 1:L])
    end
    lnZ = 0.0  # log of partition function
    for head in 1:((L÷2)-1)  # mps on the boundary is eating the next mpo, for L/2-1 times
        res, tensors[head+1] = compress(eat(tensors[head], tensors[head+1]), χ)
        lnZ += res
    end
    return 2*lnZ
end

function eat(mps, mpo)
    return [reshape(ein"ijk,abcj->iabkc"(mps[i], mpo[i]), size(mps[i], 1)*size(mpo[i], 1), 2, :) for i in 1:length(mps)]
end

function compress(mps, χ)
    residual = 0.0
    L = length(mps)
    
    # From left to right, sweep once doing QR decompositions
    for i in 1:(L-1)
        Q, R = qr(reshape(mps[i], size(mps[i], 1)*2, :))
        Q = Matrix(Q)
        mps[i] = reshape(Q, size(mps[i], 1), 2, :)
        mps[i+1] = ein"ij,jab->iab"(R, mps[i+1])
    end
    
    # From right to left, sweep once using SVD
    for i in L:-1:2
        merged = ein"ijk,kab->ijab"(mps[i-1], mps[i])
        U, S, V = svd(reshape(merged, size(mps[i-1], 1)*2, size(mps[i], 3)*2))
        
        @show size(V)
        @show V[:, 1:χ]'
        mps[i] = reshape(V[:, 1:χ]', :, 2, size(mps[i], 3))
        mps[i-1] = reshape(U[:, 1:χ] * Diagonal(S[1:χ]), size(mps[i-1], 1), 2, :)
        
        tnorm = norm(mps[i-1])
        mps[i-1] ./= tnorm
        residual += log(tnorm)
    end
    
    return residual, mps
end



# Example usage
L = 16
β_c = 0.44068679350977147
χ = 16
println("L=", L, " χ=", χ)
lnZ = get_lnZ(L)