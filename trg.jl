using OMEinsum
using LinearAlgebra

include("fib2d.jl")

function get_t_tensor(p)
    code = ein"ia,aj,ak,la,a->ijkl"

    A = Mod{p, Int}[1 1; 1 0]
    l, u = lu(A; allowsingular = true)

    one_tensor = Mod{p, Int}[1, 1]
    return code(u, l, l, u, one_tensor)
end

function trg(no_iter::Int;p = 9223372036854775783)
    T = get_t_tensor(p)
    # ld (r = l, u = d)
    bd = [2,2]

    for n in 1:no_iter
        T1 = ein"ldru -> drul"(T)
        T11 = reshape(T1, (bd[2] * bd[1], bd[2] * bd[1]))
        T2 = ein"ldru -> ldru"(T)
        T22 = reshape(T2, (bd[1] * bd[2], bd[1] * bd[2]))

        p, l, u, q = lu_with_complete_pivot(T11)
        l_new = size(l, 2)
        F1 = reshape(p*l, (bd[2], bd[1], l_new))
        F3 = reshape(u*q, (l_new, bd[2], bd[1]))
        p, l, u, q = lu_with_complete_pivot(T22)
        d_new = size(l, 2)
        F2 = reshape(p*l, (bd[1], bd[2], d_new))
        F4 = reshape(u*q, (d_new, bd[1], bd[2]))
        T = ein"(wal,abu),(rbg,dgw) -> ldru"(F1, F2, F3, F4)
        bd = [l_new, d_new]
        @show bd
    end
    # return T
    return ein"ldru,ruld -> "(T,T)[]
    return ein"ijij -> "(T)[]
end

trg(0) #  
trg(1) # 
trg(2) # 
trg(3)
trg(6)
fib2d(8,8;periodic=true)


A = Mod{p, Int}[1 1; 1 0]
one_tensor = Mod{p, Int}[1, 1]
ein"aa,aa,a-> "(A,A,one_tensor)


using Test
@testset "get_t_tensor" begin
    T = get_t_tensor(1021)
    @test size(T) == (2, 2, 2, 2)
    @test ein"ijij -> "(T)[].val == 1
end

for i in 1:10
    println("$i $(fib2d(i,i;periodic=true))")
end