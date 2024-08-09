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
scenario = NTuple{0, Symbol}()

@testset "get_case_indivdata" begin
    res = get_case_indivdata(inv_case, :A;
        scenario = CA.ComponentVector())
    # keys for different data streams
    @test all((:sv₊x, :sv₊dec2) .∈ Ref(keys(res)))
    # several information inside and same length
    #@test all((:t, :obs, :obs_true) .∈ Ref(keys(res.:sv₊x1)))
    @test length(res.:sv₊x.t) == length(res.:sv₊x.obs) == length(res.:sv₊x.obs_true)
end;

@testset "get_case_priors_dict and dict_to_cv" begin
    priors_dict = get_case_priors_dict(inv_case, :A; scenario)
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

@testset "get_case_priors_dict different for indiv" begin
    priors_dict_A = get_case_priors_dict(inv_case, :A; scenario)
    priors_dict_B = get_case_priors_dict(inv_case, :B; scenario)
    @test priors_dict_B == priors_dict_A
    scenario2 = (scenario..., :test_indiv_priors)
    priors_dict_A2 = get_case_priors_dict(inv_case, :A; scenario = scenario2)
    priors_dict_B2 = get_case_priors_dict(inv_case, :B; scenario = scenario2)
    @test priors_dict_B2[:sv₊i] != priors_dict_A2[:sv₊i]
    @test mode(priors_dict_A2[:sv₊i]) ≈ 1.0
    @test mode(priors_dict_B2[:sv₊i]) ≈ 2.0
end

@testset "setup_tools_indiv and setup_priors_pop" begin
    #popt = CA.ComponentVector(state = (sv₊x1=1.0, sv₊x2=1.0), par=(sv₊τ=1.0, sv₊i=1.0))
    popt = CA.ComponentVector(state = (sv₊x = [1.0, 1.0],),
        par = (sv₊τ = 1.0, sv₊p = fill(1.0, 3), sv₊b1 = 0.01))
    ranadd = flatten1(popt)[(:sv₊b1,)]
    ranmul = flatten1(popt)[(:sv₊x, :sv₊τ)]
    indiv = flatten1(popt)[(:sv₊p,)]
    # TODO after refactoring setup_tools_indiv
    # @test_throws "i2" setup_tools_indiv(:A; inv_case, scenario, system = sys,
    #     keys_indiv = keys(indiv))
    # res_prior = setup_tools_indiv(:A; inv_case, scenario, system = sys,
    #     keys_indiv = keys(indiv), p_default = CA.ComponentVector(sv₊i2 = 5.0,))
    # @test_throws "sv₊i2" setup_tools_indiv(:A; inv_case, scenario, system = sys,
    #     keys_indiv = keys(indiv), u0 = popt.state, p = popt.par)
    # res = setup_tools_indiv(:A; inv_case, scenario, system = sys,
    #     keys_indiv = keys(indiv), u0 = popt.state, p = popt.par, 
    #     p_default = CA.ComponentVector(sv₊i = 4.0, sv₊i2 = 5.0,))        
    # @test eltype(res.priors_indiv) <: Distribution
    # @test keys(res.priors_indiv) == keys(indiv)
    # @test get_system(res.problem) == sys
    #
    fixed = CA.ComponentVector{Float64}()
    priors_pop = setup_priors_pop(keys(fixed), keys(ranadd), keys(ranmul); inv_case, scenario)
    @test eltype(priors_pop.fixed) <: Distribution
    @test keys(priors_pop.fixed) == keys(fixed)
    #
    @test eltype(priors_pop.ranadd) <: Distribution
    @test keys(priors_pop.ranadd) == keys(ranadd)
    @test eltype(priors_pop.ranadd_σ) <: Distribution
    @test keys(priors_pop.ranadd_σ) == keys(ranadd)
    #
    @test eltype(priors_pop.ranmul) <: Distribution
    @test keys(priors_pop.ranmul) == keys(ranmul)
    @test eltype(priors_pop.ranmul_σ) <: Distribution
    @test keys(priors_pop.ranmul_σ) == keys(ranmul)
end;
