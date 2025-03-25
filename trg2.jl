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

function trg2(no_iter::Int;p = 9223372036854775783)
    T = get_t_tensor(p)
    
    # ld (r = l, u = d)
    bd = [2,2]

    for n in 1:no_iter
        T_new = ein"(abcd,cfgh),(ijkb,knof) -> aijnoghd"(T, T, T, T)
        T = reshape(T_new, (bd[1] * bd[1], bd[2] * bd[2], bd[1] * bd[1], bd[2] * bd[2]))

        bd = [bd[1] * bd[1], bd[2] * bd[2]]
        @show bd
    end
    return ein"ijij -> "(T)[]
end

trg2(0) #  
trg2(1) # 
trg2(2) # 
trg2(3)
trg(6)
fib2d(1,1;periodic=true)


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