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

mixed_keys = (;
    fixed = (:sv₊p,),
    random = (:sv₊x, :sv₊τ),
    indiv = (:sv₊i,),)

indiv_ids = (:A, :B, :C)
#p_indiv = CP.get_indiv_parameters(inv_case)
#priors_dict_indiv = get_priors_dict_indiv(inv_case, indiv_ids; scenario)    
p_indiv = get_indiv_parameters_from_priors(inv_case; scenario, indiv_ids, mixed_keys,
    system)

# get the sizes of componentVectors from prior means
# actual values are overridden below from site, after psets.opt is available
(mixed, df, psets, priors_pop, sample0) = setup_tools_mixed(p_indiv;
    inv_case, scenario, system, mixed_keys)
(fixed, random, indiv, indiv_random) = mixed

@testset "setup_psets_mixed" begin
    priors_dict_indiv = get_priors_dict_indiv(inv_case, indiv_ids; scenario)
    mean_priors_mixed = CP.mean_priors(;
        mixed_keys,
        priors_dict = first(values(priors_dict_indiv)),
        system)
    psets = setup_psets_mixed(mixed_keys; system, mean_priors_mixed.popt)
    @test all((:fixed, :random, :indiv, :popt) .∈ Ref(keys(psets)))
    (_fixed, _random, _indiv1, _popt) = mean_priors_mixed
    @test psets.fixed isa ODEProblemParSetter
    _tools = setup_tools_indiv(:A; inv_case, scenario, system,
        keys_indiv = mixed_keys.indiv,
        u0 = _popt[(:sv₊x,)], p = _popt[(:sv₊p, :sv₊τ, :sv₊i)])
    @test get_paropt_labeled(_tools.pset_u0p, _tools.problem; flat1 = Val(true)) == _popt
    @test get_paropt_labeled(psets.fixed, _tools.problem; flat1 = Val(true)) == _fixed
    @test get_paropt_labeled(psets.random, _tools.problem; flat1 = Val(true)) == _random
    @test get_paropt_labeled(psets.indiv, _tools.problem; flat1 = Val(true)) == _indiv1
    @test get_paropt_labeled(psets.popt, _tools.problem; flat1 = Val(true)) == _popt
end;

solver = AutoTsit5(Rodas5P())
sim_sols_probs = gen_sim_sols_probs(; df.tools, psets, solver)

@testset "simsols" begin
    sols_probs = sim_sols_probs(fixed, random, indiv, indiv_random)
    (sol, problem_opt) = sols_probs[1]
    popt1 = get_paropt_labeled(psets.popt, df.tools[1].problem; flat1 = Val(true))
    popt2 = get_paropt_labeled(psets.popt, problem_opt; flat1 = Val(true))
    @test popt2[keys(random)] == random .* indiv_random[:, 1] 
    @test popt2[mixed_keys.indiv] == indiv[:, 1]
    @test popt2[keys(fixed)] == fixed 
    # recomputed sites ranef and set indiv, but used mean fixed parameters
    @test popt2[keys(random)] == df.paropt[1][keys(random)]
    @test popt2[mixed_keys.indiv] == df.paropt[1][mixed_keys.indiv]
    sol = first(sols_probs).sol
    @test all(sol[sv.x][1] .== p_indiv.u0[1]) # here all state random effects
    sol[sv.x]
    sol([0.3, 0.35]; idxs = [sv.dec2, sv.dec2])
    @test all(sol([0.3, 0.35]; idxs = sv.dec2).u .> 0) # observed at a interpolated times
    solA0 = solve(df.tools[1].problem, solver)

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

tmpf = () -> begin    
    # for finding initial step size use some more adaptive steps
    chn = Turing.sample(model_cross, Turing.NUTS(1000, 0.65), n_sample,
    init_params = collect(sample0))
end

#with_logger(error_on_warning) do
chn = Turing.sample(model_cross, Turing.NUTS(n_burnin, 0.65, init_ϵ = 0.2), n_sample,
    init_params = collect(sample0))
#end

names(chn, :parameters)

# first chain as a ComponentMatrix
s1 = CA.ComponentMatrix(Array(chn[:, 1:length(sample0), 1]),
    CA.FlatAxis(), first(CA.getaxes(sample0)))
s1[:, :fixed][:, :sv₊p]

#Serialization.serialize("tmp/mixed_sample_chn.js", chn)
