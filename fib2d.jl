using GenericTensorNetworks, Graphs
using LinearAlgebra
using GenericTensorNetworks.Mods
using Random
using GenericTensorNetworks.Primes

function fib1d(n::Int)
    problem = IndependentSet(path_graph(n))
    tnet = GenericTensorNetwork(problem)
    return solve(tnet, CountingAll())[]
end

function fib2d(n::Int, m::Int; periodic=false)
    problem = IndependentSet(grid((n, m); periodic))
    tnet = GenericTensorNetwork(problem)
    return solve(tnet, CountingAll())[]
end

function fib2d_bmps(n::Int,m::Int; p = 9223372036854775783)
    tensors = get_tensors(n,m; p)
    return contract_tensors(tensors)
end

# L is the layer length
function get_tensors(L::Int, m::Int; p = 1021)
    T = eltype(p)
    # Create Boltzmann matrix
    A = Mod{p, T}[1 1; 1 0]
    l, u = lu(A; allowsingular = true)

    one_tensor = Mod{p, T}[1, 1]
    one_row = Mod{p, T}[1 1]

    code = ein"ia,aj,ak,la,a->ijkl"

    t_full = code(u, l, l, u, one_tensor)
    t_left = code(one_row, l, l, u, one_tensor)
    t_right = code(u, l, one_row', u, one_tensor)
    t_down = code(u, one_row', l, u, one_tensor)
    t_up = code(u, l, l, one_row, one_tensor)

    t_lu = code(one_row, l, l, one_row, one_tensor)
    t_ld = code(one_row, one_row', l, u, one_tensor)
    t_ru = code(u, l, one_row', one_row, one_tensor)
    t_rd = code(u, one_row', one_row', u, one_tensor)

    # Initialize tensors array
    tensors = Vector{Vector{Array{Mod{p, T}, 4}}}()
    push!(tensors, [i == 1 ? t_lu :
                    i == L ? t_ru :
                    t_up for i in 1:L])
    for j in 2:(m-1)
        push!(tensors, [i == 1 ? t_left :
                        i == L ? t_right :
                        t_full for i in 1:L])
    end
    push!(tensors, [i == 1 ? t_ld :
                    i == L ? t_rd :
                    t_down for i in 1:L])
    return tensors
end

function eat(mps, mpo)
    return [reshape(ein"ijkl,abcj->iabkcl"(s, o), size(s, 1) * size(o, 1), size(o, 2), size(s, 3) * size(o, 3), size(s, 4)) for (s, o) in zip(mps, mpo)]
end

function contract_tensors(tensors)
    L = length(tensors)
    for i in 1:(L-1)  # mps on the boundary is eating the next mpo, for L-1 times
        @show i
        tensors[i+1] = eat(tensors[i], tensors[i+1])

        # if i == 2
        #     mps2 = copy(tensors[i+1])
        #     for j in 1:(L-1)
        #         mps2[j+1] = reshape(ein"ijkl,kabc -> ijlabc"(mps2[j], mps2[j+1]), 1, :, size(mps2[j+1],3), 1)
        #     end

            compress!(tensors[i+1]; rounds = 1)
        #     mps = copy(tensors[i+1])
        #     for j in 1:(L-1)
        #         mps[j+1] = reshape(ein"ijkl,kabc -> ijlabc"(mps[j], mps[j+1]), 1, :, size(mps[j+1],3), 1)
        #     end
        #     # @show mps[end][1]
        #     # @show mps2[end][1]
        # end
    end
    for j in 1:(L-1)
        tensors[end][j+1] = reshape(ein"ijkl,kabc -> ijlabc"(tensors[end][j], tensors[end][j+1]), 1, 1, :, 1)
    end
    return tensors[end][end][]
end

function compress!(mps; rounds = 1)
    L = length(mps)
    @show size.(mps, 1)
    for _ in 1:rounds
        for i in 1:(L-1)
            p, l, u, q = lu_with_complete_pivot(reshape(mps[i], size(mps[i], 1) * size(mps[i], 2), :))
            mps[i] = reshape(p' * l, size(mps[i], 1), size(mps[i], 2), size(l, 2), size(mps[i], 4))
            mps[i+1] = ein"ij,jabc->iabc"(u * q', mps[i+1])
        end

        for i in L:-1:2
            p, l, u, q = lu_with_complete_pivot(reshape(mps[i], :, size(mps[i], 2) * size(mps[i], 3)))
            # @assert p'*l*u*q' == reshape(mps[i], :, size(mps[i], 2) * size(mps[i], 3))
            mps[i] = reshape(u * q', size(u, 1), size(mps[i], 2), size(mps[i], 3), size(mps[i], 4))
            mps[i-1] = ein"abcd,ci->abid"(mps[i-1], p' * l)
        end
        @show size.(mps, 1)
    end
    return
end

function lu_with_complete_pivot(A::AbstractMatrix{T}) where T
    n, m = size(A)

    P = Matrix{T}(I(n))
    Q = Matrix{T}(I(m))
    L = zeros(T, n, n)
    U = copy(A)


    ranka = min(n, m)
    for k in 1:min(n, m)
        # Find the pivot
        # index = argmax(abs.(U[k:end, k:end]))
        subU = view(U, k:n, k:m)
        # TODO: fix the bug in Mods
        index = findfirst(!iszero, subU)
        if isnothing(index)
            ranka = k - 1
            break
        end

        pivot_row, pivot_col = index[1], index[2]
        pivot_row += k - 1
        pivot_col += k - 1
        # Swap rows in U and P

        U[[k, pivot_row], :] = U[[pivot_row, k], :]
        L[[k, pivot_row], :] = L[[pivot_row, k], :]
        P[[k, pivot_row], :] = P[[pivot_row, k], :]

        # Swap columns in U and Q
        U[:, [k, pivot_col]] = U[:, [pivot_col, k]]
        Q[:, [k, pivot_col]] = Q[:, [pivot_col, k]]

        # Compute multipliers and update U
        for i in k+1:n
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:end] -= L[i, k] * U[k, k:end]
        end
    end

    # L[1:ranka, 1:ranka] += I(ranka)  # Add identity to L

    L += I(n) 
    return P, L[:, 1:ranka], U[1:ranka, :], Q
end