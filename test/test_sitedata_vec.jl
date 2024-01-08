using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using MTKHelpers
using OrdinaryDiffEq, ModelingToolkit
using ComponentArrays: ComponentArrays as CA
using Distributions
#using ComponentArrays

@named sv = CP.samplesystem_vec()
@named sys = embed_system(sv)
scenario = (system = :CrossInverts_samplesystem_vec,)

@testset "get_sitedata" begin
    res = get_sitedata(Val(:CrossInverts_samplesystem_vec), :A, CA.ComponentVector())
    # keys for different data streams
    @test all((:sv₊x, :sv₊dec2) .∈ Ref(keys(res)))
    # several information inside and same length
    #@test all((:t, :obs, :obs_true) .∈ Ref(keys(res.:sv₊x1)))
    @test length(res.:sv₊x.t) == length(res.:sv₊x.obs) == length(res.:sv₊x.obs_true)
end;

@testset "get_priors_df and get_priors" begin
    priors_df = get_priors_df(Val(:CrossInverts_samplesystem_vec), :A, CA.ComponentVector())
    @test all((:sv₊x, :sv₊p, :sv₊τ, :sv₊i) .∈ Ref(priors_df.par))
    @test eltype(priors_df.dist) <: Distribution
    #
    _mean = CA.ComponentVector(; zip(priors_df.par, mean.(priors_df.dist))...)
    popt = CA.ComponentVector(state = _mean[(:sv₊x,)], par = _mean[(:sv₊τ, :sv₊i)])
    priors = get_priors(keys(popt.state), priors_df)
    @test keys(priors) == keys(popt.state)
    priors = get_priors(keys(popt.par), priors_df)
    @test keys(priors) == keys(popt.par)
    priors = get_priors(reverse(keys(popt.par)), priors_df)
    @test keys(priors) == reverse(keys(popt.par))
end;

@testset "setup_tools_scenario" begin
    #popt = CA.ComponentVector(state = (sv₊x1=1.0, sv₊x2=1.0), par=(sv₊τ=1.0, sv₊i=1.0))
    popt = CA.ComponentVector(state = (sv₊x = [1.0, 1.0],),
        par = (sv₊τ = 1.0, sv₊p = fill(1.0, 3)))
    res = setup_tools_scenario(:A, scenario, popt; system = sys);
    #@test eltype(res.u_map) == eltype(res.p_map) == Int
    @test res.problemupdater isa NullProblemUpdater
    @test eltype(res.priors) <: Distribution
    @test keys(res.priors) == (keys(popt.state)..., keys(popt.par)...)
    @test axis_paropt(res.pset) == CA.getaxes(popt)[1]
    @test get_system(res.problem) == sys
end;
