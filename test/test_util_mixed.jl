using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using ModelingToolkit, OrdinaryDiffEq
using MTKHelpers
using DataFrames
using ComponentArrays: ComponentArrays as CA
using StableRNGs
using Statistics
using DistributionFits
using PDMats: PDiagMat

@named sv = CP.samplesystem_vec()
@named system = embed_system(sv)
inv_case = SampleSystemVecCase()
scenario = NTuple{0, Symbol}()

p_indiv = CP.get_indiv_parameters(inv_case)
n_indiv = nrow(p_indiv)

# get the sizes of componentVectors from prior means
# actual values are overridden below from site, after psets.opt is available
component_keys = (;
    fixed = (:sv₊p,),
    random = (:sv₊x, :sv₊τ),
    indiv = (:sv₊i,),)
priors_dictA = get_priors_dict(inv_case, first(p_indiv.indiv_id); scenario)
components_priors = CP.mean_priors(; component_keys..., priors_dict = priors_dictA, system)

# inferred from prior
toolsA_prior = setup_tools_scenario(:A; inv_case, scenario, system,
    keys_indiv = component_keys.indiv);

# setting parameters from previous run
toolsA = setup_tools_scenario(:A; inv_case, scenario, system,
    keys_indiv = component_keys.indiv, u0 = first(p_indiv.u0), p = first(p_indiv.p));

psets = setup_psets_fixed_random_indiv(keys(components_priors.fixed),
    keys(components_priors.random); system, components_priors.popt)

@testset "setup_psets_fixed_random_indiv" begin
    @test all((:fixed, :random, :indiv, :popt) .∈ Ref(keys(psets)))
    (_fixed, _random, _indiv1, _popt) = components_priors
    @test psets.fixed isa ODEProblemParSetter
    _tools = setup_tools_scenario(:A; inv_case, scenario, system,
        keys_indiv = component_keys.indiv,
        u0 = _popt[(:sv₊x,)], p = _popt[(:sv₊p, :sv₊τ, :sv₊i)])
    @test get_paropt_labeled(_tools.pset_u0p, _tools.problem; flat1 = Val(true)) == _popt
    @test get_paropt_labeled(psets.fixed, _tools.problem; flat1 = Val(true)) == _fixed
    @test get_paropt_labeled(psets.random, _tools.problem; flat1 = Val(true)) == _random
    @test get_paropt_labeled(psets.indiv, _tools.problem; flat1 = Val(true)) == _indiv1
    @test get_paropt_labeled(psets.popt, _tools.problem; flat1 = Val(true)) == _popt
end;

df = copy(p_indiv)
_extract_paropt = (u0, p) -> get_paropt_labeled(psets.popt, u0, p; flat1 = Val(true))
DataFrames.transform!(df,
    [:u0, :p] => DataFrames.ByRow(_extract_paropt) => :paropt)

(fixed, random, indiv, indiv_random) = CP.extract_mixed_effects(psets, df.paropt)

priors_pop = setup_priors_pop(keys(fixed), keys(random); inv_case, scenario);

# df = DataFrame(
#     indiv_id = collect(keys(p_indiv)), 
#     u0 = collect(map_keys(x -> x.u0, p_indiv; rewrap = Val(false))),
#     p = collect(map_keys(x -> x.p, p_indiv; rewrap = Val(false))),
# )

# popt_sites = get_paropt_labeled.(Ref(toolsA.pset), df.u0, df.p; flat1=Val(true))
# #getproperty.(popt_sites, :sv₊p)
# _tup = map(k -> mean(getproperty.(popt_sites, k)), keys(popt)) 
# popt = popt_mean = CA.ComponentVector(;zip(keys(popt), _tup)...)
# fixed = popt_mean[keys(fixed)]
# random = popt_mean[keys(random)]
# indiv = CA.ComponentMatrix(
#     hcat((popt[component_keys.indiv] for popt in popt_sites)...),
#     axis_paropt_flat1(psets.indiv), CA.Axis(df.indiv_id)
# )
# indiv_mean = popt_mean[component_keys.indiv]

# @testset "gen_compute_indiv_rand" begin
#     _compute_indiv_rand = gen_compute_indiv_rand(toolsA.pset, random)    
#     tmp = _compute_indiv_rand(toolsA.problem.u0, toolsA.problem.p)
#     @test keys(tmp) == keys(random)
#     tmp[:sv₊x] = popt_mean[:sv₊x] ./ random[:sv₊x] # popt -> toolsA -> df_site.u0/p
# end

# _compute_indiv_rand = gen_compute_indiv_rand(toolsA.pset, random)    
# DataFrames.transform!(df,
# [:u0, :p] => DataFrames.ByRow(_compute_indiv_rand) => :indiv_random)

# # extract the parameters to optimize that are individual-specific to clumns :indiv
# _extract_indiv = (u0, p) -> vcat(u0, p)[symbols_paropt(psets.indiv)]
# tmp = _extract_indiv(df.u0[1], df.p[1])
# DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_extract_indiv) => :indiv)

_setuptools = (indiv_id, u0, p) -> setup_tools_scenario(indiv_id; inv_case, scenario,
    system, u0, p, keys_indiv = component_keys.indiv);
_tools = _setuptools(df.indiv_id[1], df.u0[1], df.p[1]);
DataFrames.transform!(df, [:indiv_id, :u0, :p] => DataFrames.ByRow(_setuptools) => :tools)

solver = AutoTsit5(Rodas5P())
sim_sols_probs = gen_sim_sols_probs(; df.tools, psets, solver)

@testset "simsols" begin
    sols_probs = sim_sols_probs(fixed, random, indiv, indiv_random)
    (sol, problem_opt) = sols_probs[1]
    popt1 = get_paropt_labeled(psets.popt, df.tools[1].problem; flat1 = Val(true))
    popt2 = get_paropt_labeled(psets.popt, problem_opt; flat1 = Val(true))
    @test popt2[keys(random)] == random .* indiv_random[:, 1] #df.paropt[first(keys(df.paropt[1]))][keys(random)]
    @test popt2[component_keys.indiv] == indiv[:, 1]
    @test popt2[keys(fixed)] == fixed # popt_sites[first(keys(popt_sites))][keys(fixed)]
    # recomputed sites ranef and set indiv, but used mean fixed parameters
    @test popt2[keys(random)] == df.paropt[1][keys(random)]
    @test popt2[component_keys.indiv] == df.paropt[1][component_keys.indiv]
    sol = first(sols_probs).sol
    @test all(sol[sv.x][1] .== p_indiv.u0[1]) # here all state random effects
    sol[sv.x]
    sol([0.3, 0.35]; idxs = [sv.dec2, sv.dec2])
    @test all(sol([0.3, 0.35]; idxs = sv.dec2).u .> 0) # observed at a interpolated times
    solA0 = solve(df.tools[1].problem, solver)
    #solA0 == sol # random and individual parameters for original problem at mean instead of indiv_id

    sim_sols = gen_sim_sols(; tools = df.tools, psets, solver, maxiters = 1e4)
    poptl = (; fixed, random, indiv, indiv_random,)
    sols = sim_sols(poptl)
    sol2 = first(sols)
    @test sol2.t == sol.t
    @test sol2[sv.x] == sol[sv.x]
    #@test sol2 == sol # Stackoverflow on older versions
end;

# @testset "extract_stream_obsmatrices" begin
#     tools = tools
#     vars = (:obs, :obs_true)
#     obs = CP.extract_stream_obsmatrices(;tools, vars)
#     @test keys(obs) == (:sv₊x, :sv₊dec2) # streams
#     @test all( (keys(obs[k]) == (:t, vars...) for k in keys(obs)) ) # keys in each stream
#     #
#     # inconsistent times
#     tools2 = deepcopy(tools)
#     tools2[1].sitedata.sv₊x.t[1] = 0.222
#     @test_throws (ErrorException) CP.extract_stream_obsmatrices(;tools = tools2, vars)
#     #
#     # inconsistent length of observations
#     tools2 = deepcopy(tools)
#     tools2[1] = merge(tools2[1],
#         (;
#             sitedata = merge(tools2[1].sitedata,
#                 (;
#                     sv₊x = merge(tools2[1].sitedata.sv₊x,
#                         (; obs = tools2[1].sitedata.sv₊x.obs[2:end]))))))
#     @test_throws (ErrorException) CP.extract_stream_obsmatrices(; tools = tools2, vars)
#     #
#     # consistent observations but inconsistent with time
#     tools2 = deepcopy(tools)
#     tools2_i = tools2[1]
#     tools2 = [merge(tools2_i,
#         (;
#             sitedata = merge(tools2_i.sitedata,
#                 (;
#                     sv₊x = merge(tools2_i.sitedata.sv₊x,
#                         (; obs = tools2_i.sitedata.sv₊x.obs[2:end]))))))
#               for tools2_i in tools2]
#     #[t.sitedata.sv₊x.obs for t in tools2] # just to check that each indiv_id has only 3 obs
#     @test_throws (ErrorException) CP.extract_stream_obsmatrices(; tools = tools2, vars)
# end

using Turing
n_burnin = 0
n_sample = 10

priors_σ = CP.get_priors_random_dict(inv_case; scenario)
sample0 = CP.get_init_mixedmodel(psets, df.paropt, priors_σ; indiv_ids = df.indiv_id)

using Logging, LoggingExtras
error_on_warning = EarlyFilteredLogger(global_logger()) do log_args
    if log_args.level >= Logging.Warn
        error(log_args)
    end
    return true
end;

model_cross = gen_model_cross(;
    inv_case, tools = df.tools, priors_pop, psets, sim_sols_probs, scenario, solver);

#with_logger(error_on_warning) do
chn = Turing.sample(model_cross, Turing.NUTS(n_burnin, 0.65, init_ϵ = 1e-2), n_sample,
    init_params = collect(sample0))
#end

names(chn, :parameters)

# first chain as a ComponentMatrix
s1 = CA.ComponentMatrix(Array(chn[:, 1:length(sample0), 1]),
    CA.FlatAxis(), first(CA.getaxes(sample0)))
s1[:, :fixed][:, :sv₊p]
