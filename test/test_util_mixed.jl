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
priors_dict_indiv = Dict(id => get_priors_dict(inv_case, id; scenario) for id in p_indiv.indiv_id)


# get the sizes of componentVectors from prior means
# actual values are overridden below from site, after psets.opt is available
component_keys = (;
    fixed = (:sv₊p,),
    random = (:sv₊x, :sv₊τ),
    indiv = (:sv₊i,),)
(mixed, df, psets, priors_pop, sample0)  = setup_tools_mixed(p_indiv, priors_dict_indiv;
    inv_case, scenario, system, component_keys)
(fixed, random, indiv, indiv_random) = mixed


tmpf = () -> begin 
    # inferred from prior
    toolsA_prior = setup_tools_indiv(:A; inv_case, scenario, system,
        keys_indiv = component_keys.indiv);
    # setting parameters from previous run
    toolsA = setup_tools_indiv(:A; inv_case, scenario, system,
        keys_indiv = component_keys.indiv, u0 = first(p_indiv.u0), p = first(p_indiv.p));
end


@testset "setup_psets_fixed_random_indiv" begin
    mean_priors_mixed = CP.mean_priors(; component_keys..., priors_dict = first(values(priors_dict_indiv)), system)
    psets = setup_psets_fixed_random_indiv(keys(mean_priors_mixed.fixed),
        keys(mean_priors_mixed.random); system, mean_priors_mixed.popt)
    @test all((:fixed, :random, :indiv, :popt) .∈ Ref(keys(psets)))
    (_fixed, _random, _indiv1, _popt) = mean_priors_mixed
    @test psets.fixed isa ODEProblemParSetter
    _tools = setup_tools_indiv(:A; inv_case, scenario, system,
        keys_indiv = component_keys.indiv,
        u0 = _popt[(:sv₊x,)], p = _popt[(:sv₊p, :sv₊τ, :sv₊i)])
    @test get_paropt_labeled(_tools.pset_u0p, _tools.problem; flat1 = Val(true)) == _popt
    @test get_paropt_labeled(psets.fixed, _tools.problem; flat1 = Val(true)) == _fixed
    @test get_paropt_labeled(psets.random, _tools.problem; flat1 = Val(true)) == _random
    @test get_paropt_labeled(psets.indiv, _tools.problem; flat1 = Val(true)) == _indiv1
    @test get_paropt_labeled(psets.popt, _tools.problem; flat1 = Val(true)) == _popt
end;





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


using Turing
n_burnin = 0
n_sample = 10


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

