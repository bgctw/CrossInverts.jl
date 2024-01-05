using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using MTKHelpers
using OrdinaryDiffEq, ModelingToolkit
using ComponentArrays: ComponentArrays as CA
using Distributions
#using ComponentArrays

@named mv = CP.samplesystem_vec()
@named sys = embed_system(mv)
scenario = (system = :CrossInverts_samplesystem_vec,)

@testset "get_sitedata" begin
    res = get_sitedata(Val(:CrossInverts_samplesystem_vec), :A, CA.ComponentVector())
    # keys for different data streams
    @test all((:mv₊x, :mv₊dec2) .∈ Ref(keys(res)))
    # several information inside and same length
    #@test all((:t, :obs, :obs_true) .∈ Ref(keys(res.:mv₊x1)))
    @test length(res.:mv₊x.t) == length(res.:mv₊x.obs) == length(res.:mv₊x.obs_true)
end;

@testset "get_priors_df and get_priors" begin
    priors_df = get_priors_df(Val(:CrossInverts_samplesystem_vec), :A, CA.ComponentVector())
    @test all((:mv₊x, :mv₊p, :mv₊τ, :mv₊i) .∈ Ref(priors_df.par))
    @test eltype(priors_df.dist) <: Distribution
    #
    _mean = CA.ComponentVector(; zip(priors_df.par, mean.(priors_df.dist))...)
    popt = CA.ComponentVector(state = _mean[(:mv₊x,)], par = _mean[(:mv₊τ, :mv₊i)])
    priors = get_priors(keys(popt.state), priors_df)
    @test keys(priors) == keys(popt.state)
    priors = get_priors(keys(popt.par), priors_df)
    @test keys(priors) == keys(popt.par)
    priors = get_priors(reverse(keys(popt.par)), priors_df)
    @test keys(priors) == reverse(keys(popt.par))
end;

@testset "setup_tools_scenario" begin
    #popt = CA.ComponentVector(state = (mv₊x1=1.0, mv₊x2=1.0), par=(mv₊τ=1.0, mv₊i=1.0))
    popt = CA.ComponentVector(state = (mv₊x = [1.0, 1.0],),
        par = (mv₊τ = 1.0, mv₊p = fill(1.0, 3)))
    res = setup_tools_scenario(:A, scenario, popt; system = sys);
    #@test eltype(res.u_map) == eltype(res.p_map) == Int
    @test res.problemupdater isa NullProblemUpdater
    @test eltype(res.priors) <: Distribution
    @test keys(res.priors.state) == keys(popt.state)
    @test keys(res.priors.par) == keys(popt.par)
    @test axis_paropt(res.pset) == CA.getaxes(popt)[1]
    @test get_system(res.problem) == sys
end;
