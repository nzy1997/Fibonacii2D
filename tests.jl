using Test

@testset "lu_with_complete_pivot" begin
    Random.seed!(1234)
    p = 1021
    A = rand(Mod{p, Int}, 5, 4)
    P, L, U, Q = lu_with_complete_pivot(A)
    display(L)
    display(U)
    Aa = P * L * U * Q
    @test Aa == A

    A = Mod{5, Int32}[0 1 1 1 4 1 0;0 1 4 1 2 3 3;0 4 0 1 0 1 0;0 0 0 0 0 0 0;0 1 1 3 3 2 2;0 3 3 0 0 4 4;0 0 0 0 0 0 0;0 0 0 0 0 0 0]
    P, L, U, Q = lu_with_complete_pivot(A)
    display(L)
    display(U)
    Aa = L * U 
    @test Aa == P * A* Q
end

@testset "get_tensors" begin
    L = 3
    tensors = get_tensors(L)
    for i in 1:(L-1)  # mps on the boundary is eating the next mpo, for L-1 times
        tensors[i+1] = eat(tensors[i], tensors[i+1])
    end

    @test ein"ijkl,kabc,bdef -> ijlacdef"(tensors[end]...)[] == 63
end

@testset "lu_real" begin
    Random.seed!(1234)
    A = rand(Float64, 5, 4)
    P, L, U, Q = lu_with_complete_pivot(A)
    Aa = P * L * U * Q
    @test Aa ≈ A atol = 1e-10
end