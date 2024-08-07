using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using MTKHelpers
using OrdinaryDiffEq, ModelingToolkit
using ComponentArrays: ComponentArrays as CA
using SymbolicIndexingInterface: SymbolicIndexingInterface as SII

@named m2 = CP.samplesystem_vec()
@named sys = embed_system(m2)

@testset "system with symbolic arrays" begin
    st = vcat(Symbolics.scalarize(m2.x .=> [1.0, 2.0]))
    p_new = vcat(Symbolics.scalarize(m2.p .=> [2.1, 2.2, 2.3]), m2.i2 => 5.0)
    prob = ODEProblem(sys, st, (0.0, 10.0), p_new)
    @test SII.getp(sys, :m2₊τ)(prob) == 3.0
    @test SII.getp(sys, :m2₊i)(prob) == 0.1
    @test SII.getp(sys, m2.p)(prob) == [2.1, 2.2, 2.3]
    sol = solve(prob, Tsit5())
    # last to access the second value of the pair
    @test first(sol[m2.x]) == last.(st[1:2])
    #sol[m2.x[1]]
    #plot(sol, vars=[m2.x,m2.RHS])    
    #
    # specify by symbol_op instead of num
    _dict_nums = get_system_symbol_dict(sys)
    st = vcat(Symbolics.scalarize(_dict_nums[:m2₊x] .=> [10.1, 10.2]))
    prob = ODEProblem(sys, st, (0.0, 10.0),
        [_dict_nums[:m2₊τ] => 3.0, _dict_nums[:m2₊i2] => 5.0])
    @test prob.u0 == [10.1, 10.2]
end;
