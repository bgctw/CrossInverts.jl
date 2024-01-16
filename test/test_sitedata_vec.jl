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
inv_case = SampleSystemVecCase()
scenario = NTuple{0,Symbol}()

@testset "get_sitedata" begin
    res = get_sitedata(inv_case, :A; 
        scenario = CA.ComponentVector())
    # keys for different data streams
    @test all((:sv₊x, :sv₊dec2) .∈ Ref(keys(res)))
    # several information inside and same length
    #@test all((:t, :obs, :obs_true) .∈ Ref(keys(res.:sv₊x1)))
    @test length(res.:sv₊x.t) == length(res.:sv₊x.obs) == length(res.:sv₊x.obs_true)
end;

@testset "get_priors_dict and dict_to_cv" begin
    priors_dict = get_priors_dict(inv_case, :A; scenario)
    @test all((:sv₊x, :sv₊p, :sv₊τ, :sv₊i) .∈ Ref(keys(priors_dict)))
    @test eltype(values(priors_dict)) <: Distribution
    #
    _mean = CA.ComponentVector(; zip(keys(priors_dict), mean.(values(priors_dict)))...)
    popt = CA.ComponentVector(state = _mean[(:sv₊x,)], par = _mean[(:sv₊τ, :sv₊i)])
    priors = dict_to_cv(keys(popt.state), priors_dict)
    @test keys(priors) == keys(popt.state)
    priors = dict_to_cv(keys(popt.par), priors_dict)
    @test keys(priors) == keys(popt.par)
    priors = dict_to_cv(reverse(keys(popt.par)), priors_dict)
    @test keys(priors) == reverse(keys(popt.par))
end;

@testset "setup_tools_scenario and setup_priors_pop" begin
    #popt = CA.ComponentVector(state = (sv₊x1=1.0, sv₊x2=1.0), par=(sv₊τ=1.0, sv₊i=1.0))
    popt = CA.ComponentVector(state = (sv₊x = [1.0, 1.0],),
        par = (sv₊τ = 1.0, sv₊p = fill(1.0, 3)))
    random = flatten1(popt)[(:sv₊x, :sv₊τ)]
    indiv = flatten1(popt)[(:sv₊p,)] 
    res = setup_tools_scenario(:A; inv_case, scenario, popt, system = sys, keys_indiv = keys(indiv));
    #@test eltype(res.u_map) == eltype(res.p_map) == Int
    @test res.problemupdater isa NullProblemUpdater
    @test eltype(res.priors_indiv) <: Distribution
    @test keys(res.priors_indiv) == keys(indiv)
    @test axis_paropt(res.pset) == CA.getaxes(popt)[1]
    @test get_system(res.problem) == sys
    #
    fixed = CA.ComponentVector{Float64}()
    priors_pop = setup_priors_pop(keys(fixed), keys(random); inv_case, scenario);
    @test eltype(priors_pop.fixed) <: Distribution
    @test keys(priors_pop.fixed) == keys(fixed)
    @test eltype(priors_pop.random) <: Distribution
    @test keys(priors_pop.random) == keys(random)
    @test eltype(priors_pop.random_σ) <: Distribution
    @test keys(priors_pop.random_σ) == keys(random)
end;
