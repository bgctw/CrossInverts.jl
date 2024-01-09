using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using MTKHelpers
using OrdinaryDiffEq, ModelingToolkit
using ComponentArrays: ComponentArrays as CA
using Distributions
#using ComponentArrays

@named m1 = CP.samplesystem1()
@named sys = embed_system(m1)
scenario = (system = :CrossInverts_samplesystem1,)

@testset "get_sitedata" begin
    res = get_sitedata(Val(:CrossInverts_samplesystem1), :A, CA.ComponentVector())
    # keys for different data streams
    @test all((:m1₊x1, :m1₊dec2) .∈ Ref(keys(res)))
    # several information inside and same length
    #@test all((:t, :obs, :obs_true) .∈ Ref(keys(res.:m1₊x1)))
    @test length(res.:m1₊x1.t) == length(res.:m1₊x1.obs) == length(res.:m1₊x1.obs_true)
end;

@testset "get_priors_dict and dict_to_cv" begin
    priors_dict = get_priors_dict(Val(:CrossInverts_samplesystem1), :A, CA.ComponentVector())
    @test all((:m1₊x1, :m1₊x2, :m1₊τ, :m1₊i) .∈ Ref(keys(priors_dict)))
    @test eltype(values(priors_dict)) <: Distribution
    #
    _mean = CA.ComponentVector(; zip(keys(priors_dict), mean.(values(priors_dict)))...)
    popt = CA.ComponentVector(state = _mean[(:m1₊x1, :m1₊x2)], par = _mean[(:m1₊τ, :m1₊i)])
    priors = dict_to_cv(keys(popt.state), priors_dict)
    @test keys(priors) == keys(popt.state)
    priors = dict_to_cv(keys(popt.par), priors_dict)
    @test keys(priors) == keys(popt.par)
    priors = dict_to_cv(reverse(keys(popt.par)), priors_dict)
    @test keys(priors) == reverse(keys(popt.par))
end;

@testset "setup_tools_scenario" begin
    #popt = CA.ComponentVector(state = (m1₊x1=1.0, m1₊x2=1.0), par=(m1₊τ=1.0, m1₊i=1.0))
    popt = CA.ComponentVector(state = (m1₊x1 = 1.0, m1₊x2 = 1.0),
        par = (m1₊τ = 1.0, m1₊p = fill(1.0, 3)))
    res = setup_tools_scenario(:A, scenario, popt; system = sys)
    #@test eltype(res.u_map) == eltype(res.p_map) == Int
    @test res.problemupdater isa NullProblemUpdater
    @test eltype(res.priors) <: Distribution
    @test keys(res.priors) == (keys(popt.state)..., keys(popt.par)...)
    @test axis_paropt(res.pset) == CA.getaxes(popt)[1]
    @test get_system(res.problem) == sys
end;
