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
using Turing
using Logging, LoggingExtras

inv_case = SampleSystemVecCase()
scenario = NTuple{0, Symbol}()

tmpf = () -> begin
    @named sv = CP.samplesystem_vec()
    @named system = embed_system(sv)
    #(;system, u0_default, p_default) = get_inverted_system(inv_case; scenario)

    mixed_keys = (;
        fixed = (:sv₊p,),
        random = (:sv₊x, :sv₊τ),
        indiv = (:sv₊i,))

    indiv_ids = (:A, :B, :C)

    #p_indiv = CP.get_indiv_parameters(inv_case)
    #priors_dict_indiv = get_priors_dict_indiv(inv_case, indiv_ids; scenario)    
    p_indiv = CP.get_indiv_parameters_from_priors(inv_case; scenario, indiv_ids, mixed_keys,
        system_u0_p_default = (; system,
            u0_default = CA.ComponentVector(), p_default = CA.ComponentVector(sv₊i2 = 0.1)))

    # get the sizes of ComponentVectors from prior means
    # actual values are overridden below from site, after psets.opt is available
    (; mixed, indiv_info, pop_info) = setup_tools_mixed(p_indiv;
        inv_case, scenario, system, mixed_keys)
end

@testset "setup_indiv_problems" begin
    tspans = fill((0.0, 0.0), 3)
    _problems = CP.setup_indiv_problems(; inv_case, scenario, tspans)
    pset = ODEProblemParSetter(get_system(first(_problems)), Symbol[])
    p = get_par_labeled(pset, _problems[1])
    @test p[:sv₊τ] == 1.5     # from get_u0p
    @test isfinite(p[:sv₊i])  # from get_priors
    @test p[:sv₊i2] == 0.1    # from get_inverted_system
    u = get_state_labeled(pset, _problems[1])
    @test u[:sv₊x] == [2.0, 2.0] # grom getu0p
    p3 = get_par_labeled(pset, _problems[3]) # no information in u0p 
    @test all(isfinite.(p3))
    @test keys(p3) == keys(p)
    u3 = get_state_labeled(pset, _problems[3])
    @test all(isfinite.(u3))
end;

tmpf = () -> begin
    system_u0_p_default = get_inverted_system(inv_case; scenario)
    (; system, u0_default, p_default) = system_u0_p_default
    (; system, pop_info, indiv_info) = setup_inversion(
        inv_case; scenario, system_u0_p_default)
end

(; system, indiv_info, pop_info) = setup_inversion(inv_case; scenario)

(; fixed, random, indiv, indiv_random) = pop_info.mixed
(; psets, problemupdater, priors_pop, sample0, effect_pos) = pop_info

@testset "setup_tools_mixed" begin
    @test problemupdater isa ProblemUpdater
    @test problemupdater.pget.source_keys == (:sv₊i,)
end;

@testset "setup_psets_mixed" begin
    indiv_ids = pop_info.indiv_ids
    mixed_keys = pop_info.mixed_keys
    priors_dict_indiv = get_priors_dict_indiv(inv_case, indiv_ids; scenario)
    mean_priors_mixed = CP.mean_priors(;
        mixed_keys,
        priors_dict = first(values(priors_dict_indiv)),
        system)
    psets = setup_psets_mixed(mixed_keys; system)
    @test all((:fixed, :random, :indiv, :popt) .∈ Ref(keys(psets)))
    (_fixed, _random, _indiv1, _popt) = mean_priors_mixed
    @test psets.fixed isa ODEProblemParSetter
    _tools = setup_tools_indiv(:A; inv_case, scenario, system,
        keys_indiv = mixed_keys.indiv,
        u0 = _popt[filter(∈(keys_state(psets.fixed)), keys(_popt))],
        p = _popt[filter(∈(keys_par(psets.fixed)), keys(_popt))],
        p_default = CA.ComponentVector(sv₊i2 = 0.1)
    )
    @test flatten1(get_paropt_labeled(psets.fixed, _tools.problem)) ==
          _popt[keys_paropt(psets.fixed)]
    @test flatten1(get_paropt_labeled(psets.fixed, _tools.problem)) == _fixed
    @test flatten1(get_paropt_labeled(psets.random, _tools.problem)) == _random
    @test flatten1(get_paropt_labeled(psets.indiv, _tools.problem)) == _indiv1
    @test flatten1(get_paropt_labeled(psets.popt, _tools.problem)) == _popt
end;

solver = AutoTsit5(Rodas5P())
sim_sols_probs = gen_sim_sols_probs(;
    tools = indiv_info.tools, psets = pop_info.psets,
    problemupdater = pop_info.problemupdater, solver)

@testset "simsols" begin
    mixed_keys = pop_info.mixed_keys
    sols_probs = sim_sols_probs(fixed, random, indiv, indiv_random)
    (sol, problem_opt) = sols_probs[1]
    popt1 = flatten1(get_paropt_labeled(psets.popt, indiv_info.tools[1].problem))
    popt2 = flatten1(get_paropt_labeled(psets.popt, problem_opt))
    @test popt2[keys(random)] == random .* indiv_random[:, 1]
    @test popt2[mixed_keys.indiv] == indiv[:, 1]
    @test popt2[keys(fixed)] == fixed
    # recomputed sites ranef and set indiv, but used mean fixed parameters
    @test popt2[keys(random)] == flatten1(indiv_info.paropt[1])[keys(random)]
    @test popt2[mixed_keys.indiv] == flatten1(indiv_info.paropt[1])[mixed_keys.indiv]
    sol = first(sols_probs).sol
    sv = system.sv
    @test all(sol[sv.x][1] .== indiv_info.u0[1]) # here all state random effects
    sol[sv.x]
    sol([0.3, 0.35]; idxs = [sv.dec2, sv.dec2])
    @test all(sol([0.3, 0.35]; idxs = sv.dec2).u .> 0) # observed at a interpolated times
    solA0 = solve(indiv_info.tools[1].problem, solver)
    # check that Problemupdater also pdated i2 to optimized i
    @test get_par_labeled(psets.fixed, problem_opt)[:sv₊i2] ==
          get_par_labeled(psets.fixed, problem_opt)[:sv₊i]
    @test get_par_labeled(psets.fixed, problem_opt)[:sv₊i2] ≠
          get_par_labeled(psets.fixed, indiv_info.tools[1].problem)[:sv₊i2]
    #
    sim_sols = gen_sim_sols(;
        tools = indiv_info.tools, psets = pop_info.psets,
        problemupdater = pop_info.problemupdater, solver, maxiters = 1e4)
    poptl = (; fixed, random, indiv, indiv_random)
    sols = sim_sols(poptl)
    sol2 = first(sols)
    @test sol2.t == sol.t
    @test sol2[sv.x] == sol[sv.x]
    #@test sol2 == sol # Stackoverflow on older versions
end;

n_burnin = 0
n_sample = 10

error_on_warning = EarlyFilteredLogger(global_logger()) do log_args
    if log_args.level >= Logging.Warn
        error(log_args)
    end
    return true
end;

model_cross = gen_model_cross(;
    inv_case, tools = indiv_info.tools,
    priors_pop = pop_info.priors_pop,
    psets = pop_info.psets,
    sim_sols_probs, scenario, solver);

tmpf = () -> begin
    # for finding initial step size use some more adaptive steps
    chn = Turing.sample(model_cross, Turing.NUTS(1000, 0.65), n_sample,
        init_params = collect(sample0))
end

#with_logger(error_on_warning) do
chn = Turing.sample(model_cross, Turing.NUTS(n_burnin, 0.65, init_ϵ = 0.2),
    MCMCThreads(), n_sample, 2, init_params = collect(sample0))
#end

tmpf = () -> begin
    names(chn, :parameters)
    # first chain as a ComponentMatrix
    s1 = CA.ComponentMatrix(Array(chn[:, 1:length(sample0), 1]),
        CA.FlatAxis(), first(CA.getaxes(sample0)))
    s1[:, :fixed][:, :sv₊p]
end

@testset "Turing indices match sample" begin
    chn2 = chn[:, vcat(effect_pos[:indiv_random][:B]...), :]
    @test names(chn2) == [Symbol("indiv_random[:sv₊x, 2][1]"),
        Symbol("indiv_random[:sv₊x, 2][2]"), Symbol("indiv_random[:sv₊τ, 2]")]
end;

tmpf = () -> begin
    #experiment with accessing subgroups
    names(chn)
    sections(chn)
    namesingroup(chn, :random)
    namesingroup(chn, :prand_σ)
    namesingroup(chn, :fixed)
    namesingroup(chn, :indiv)

    namesingroup(chn, :indiv_random)

    sample0.indiv_random

    tmp = group(chn, :random)
    tmp = group(chn, :indiv)
    replacenames(tmp, "indiv[:sv₊i, 1]" => "sv₊i[:A]")
    sample0[:indiv]

    chn = group(chn, :fixed)
    keys(sample0.fixed)

    tmp = extract_group(chn, :fixed)
    tmp = extract_group(chn, :prand_σ)
    tmp = extract_group(chn, :indiv)
    tmp = extract_group(chn, :indiv, indiv_ids)
    tmp = extract_group(chn, :indiv_random)
    tmp = extract_group(chn, :indiv_random, indiv_ids)
    tmp = compute_indiv_random(chn, indiv_ids)

    replace(":sv₊i, 1", r",\s*1$" => "[X]")
    arr = cat(Array(chn, append_chains = false)..., dims = 3)
    tmp = CA.ComponentArray(arr, CA.FlatAxis(), first(CA.getaxes(sample0)), CA.FlatAxis())

    tmp[:, :fixed, :]
    chn2 = get_example_chain()
end