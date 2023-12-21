using CrossInverts
using Test
using Aqua

@testset "CrossInverts.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(CrossInverts)
    end
    # Write your tests here.
end
