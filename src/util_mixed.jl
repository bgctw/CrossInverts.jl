"""
Given `inv_case`, the keys for different mixed effects, individual state (`u0`, `p`), 
and individual priors, then setup
- `mixed`: mixed effects NamedTuple(fixed, random, indiv, indiv_random)
  from individual's states and parameters
- `df`: DataFrame `p_indiv` extended by columns
  - `paropt`: optimized parameters extracted from indiviudals state and parameters
  - `tools`: tools initialized for each site (see `setup_tools_indiv`)
- `psets`: `NTuple{ODEProblemParSetter}` for each mixed component
- `priors_pop`: `ComponentVector` of priors on population level (fixed, random, random_σ)
- `sample0`: ComponentVector of an initial sample
  This can be used to name (attach_axis) a sample from MCMCChains object 
"""
function setup_tools_mixed(p_indiv, priors_dict_indiv;
        inv_case, scenario, system, mixed_keys,
        psets = setup_psets_mixed(inv_case; scenario, mixed_keys, system))
    df = copy(p_indiv)
    _extract_paropt = (u0, p) -> get_paropt_labeled(psets.popt, u0, p; flat1 = Val(true))
    DataFrames.transform!(df,
        [:u0, :p] => DataFrames.ByRow(_extract_paropt) => :paropt)
    mixed = extract_mixed_effects(psets, df.paropt)
    priors_pop = setup_priors_pop(keys(mixed.fixed), keys(mixed.random); inv_case, scenario)
    _setuptools = (indiv_id, u0, p) -> setup_tools_indiv(indiv_id; inv_case, scenario,
        system, u0, p, keys_indiv = mixed_keys.indiv)
    #_tools = _setuptools(df.indiv_id[1], df.u0[1], df.p[1])
    DataFrames.transform!(df,
        [:indiv_id, :u0, :p] => DataFrames.ByRow(_setuptools) => :tools)
    sample0 = get_init_mixedmodel(psets,
        df.paropt,
        priors_pop.random_σ;
        indiv_ids = df.indiv_id)
    (; mixed, df, psets, priors_pop, sample0)
end

# """
#     gen_compute_indiv_rand(pset, random)

# Generate a function closure `compute_indiv_rand(u0, p)` that for each population
# random effect-mean `random` computes the individual random effect.
# It uses the provided `ProblemParSetter` to extract the optimized parameters
# from u0 and p.
# It is used to get an initial estimate of the random effects given a population
# mean, and the individual indiv_id parameters.
# """
# function gen_compute_indiv_rand(pset::AbstractProblemParSetter, random)
#     let pset = pset, random = random
#         compute_indiv_rand = (u0, p) -> begin
#             local popt = get_paropt_labeled(pset, u0, p; flat1 = Val(true))
#             # k = first(keys(random))
#             gen = (popt[k] ./ random[k] for k in keys(random))
#             v = reduce(vcat, gen)
#             MTKHelpers.attach_axis(v, first(getaxes(random)))
#         end
#     end # let
# end

"""
    setup_psets_fixed_random_indiv(fixed, random; system, popt)

Setup the ProblemParSetters for given system and parameter names.
Assume, that parameters are fiven in flat format, i.e. not state and par labels.
Make sure, that popt holds state entries first.

Only the entries in fixed and random that are actually occurring
in popt_names are set.
The indiv parameters are the difference set between popt_names and the others.

Returns a NamedTuple with `ODEProblemParSetter` for `fixed`, `random`, and `indiv` parameters.
"""
function setup_psets_fixed_random_indiv(keys_fixed, keys_random; system, popt)
    fixed1 = intersect(keys(popt), keys_fixed)
    random1 = intersect(keys(popt), keys_random)
    indiv1 = setdiff(keys(popt), union(fixed1, random1))
    psets = (;
        fixed = ODEProblemParSetter(system, popt[fixed1]),
        random = ODEProblemParSetter(system, popt[random1]),
        indiv = ODEProblemParSetter(system, popt[indiv1]),
        popt = ODEProblemParSetter(system, popt),)
    # k = :state
    # cvs = (popt,fixed,random)
    # tup = map(keys(popt)) do k
    #     keys_k = (;zip((:popt, :fixed, :random), map(x -> haskey(x,k) ? keys(x[k]) : NTuple{0,Symbol}(), cvs))...)
    #     fixed1 = intersect(keys_k.popt, keys_k.fixed)
    #     random1 = intersect(keys_k.popt, keys_k.random)
    #     opt_site1 = setdiff(keys_k.popt, union(fixed1, random1))
    #     popt[k][opt_site1]
    #     #isempty(opt_site1) ? ComponentVector() : opt_site1
    # end
    # indiv = ComponentVector(;zip(keys(popt),tup)...)
    # psets = (;
    #     fixed = ODEProblemParSetter(system, Axis(fixed1)),
    #     random = ODEProblemParSetter(system, Axis(random1)),
    #     indiv = ODEProblemParSetter(system, Axis(opt_site1)))
    # psets = (;
    #     fixed = ODEProblemParSetter(system, fixed),
    #     random = ODEProblemParSetter(system, random),
    #     indiv = ODEProblemParSetter(system, indiv))
    return psets
end

function setup_psets_mixed(inv_case::AbstractCrossInversionCase;
        scenario, mixed_keys, system,
        priors_dict = get_priors_dict(inv_case, missing; scenario),)
    mean_priors_mixed = mean_priors(; mixed_keys..., priors_dict, system)
    setup_psets_mixed(mixed_keys; system, mean_priors_mixed.popt)
end
function setup_psets_mixed(mixed_keys; system, popt)
    gen = ((kc, ODEProblemParSetter(system, popt[mixed_keys[kc]])) for
           kc in keys(mixed_keys))
    psets = (; gen..., popt = ODEProblemParSetter(system, popt),)
    return psets
end

# function setup_psets_fixed_random_indiv(system, chn::MCMCChains.Chains)
#     @error "not fully implemented yet."
#     opt = extract_popt_names(chn)
#     setup_psets_fixed_random_indiv(system,
#         opt[(:par_names, :fixed, :random)]...)
# end

"""
    sim_sols_probs(fixed, random, indiv, indiv_random)

Update and simulate system (given with tools to gen_sim_sols_probs) by 
- for each individual i
  - update fixed parameters: fixed 
  - update random parameters: random .* indiv_random[:,i]
  - update indiv_id parameters: indiv[:,i]
  - simulate the problem
- return a vector(n_indiv) of (;sol, problem_opt)

If non-optimized p and u0 differ between individuals, they must already be
set in tools[i_indiv].problem.
"""
function gen_sim_sols_probs(; tools, psets, solver = AutoTsit5(Rodas5()), kwargs_gen...)
    fLogger = EarlyFilteredLogger(current_logger()) do log
        #@show log
        !(log.level == Logging.Warn && log.group == :integrator_interface)
    end
    n_indiv = length(tools)
    # see help on MTKHelpers.ODEProblemParSetterConcrete
    fbarrier = (psets = map(get_concrete, psets),
    problemupdater = get_concrete(first(tools).problemupdater)) -> let solver = solver,
        problems_indiv = map(t -> t.problem, tools),
        psets = psets, problemupdater = problemupdater,
        n_indiv = n_indiv,
        kwargs_gen = kwargs_gen,
        fLogger = fLogger
        #, psetci_u_PlantNP=psetci_u_PlantNP 
        kwargs_indiv_default = fill((), n_indiv)
        #
        (fixed, random, indiv, indiv_random; kwargs_indiv = kwargs_indiv_default, kwargs...) -> begin
            map(1:n_indiv) do i_indiv
                problem_opt = problems_indiv[i_indiv] # no need to copy, in update_statepar 
                problem_opt = @inferred remake(problem_opt, fixed, psets.fixed)
                problem_opt = remake(problem_opt,
                    random .* indiv_random[:, i_indiv], psets.random)
                problem_opt = remake(problem_opt, indiv[:, i_indiv], psets.indiv)
                # make sure to use problemupdater on the last update
                problem_opt = @inferred problemupdater(problem_opt)
                #if !isempty(kwargs); problem_opt = remake(problem_opt; kwargs...); end
                #local parl = label_par(psetci, problem_opt.p) #@inferred label_par(psetci, problem_opt.p)
                #suppress does not work with MCMCThreads() sampling
                #sol = @suppress solve(problem_opt, solver, maxiters = 1e4)
                sol = with_logger(fLogger) do
                    sol = solve(problem_opt, solver;
                        kwargs_gen..., kwargs_indiv[i_indiv]..., kwargs...)
                end
                (; sol, problem_opt)
            end # map 1:n_indiv
        end # function  
    end # let
    sim_sols_probs = fbarrier()
    return sim_sols_probs
end

function gen_sim_sols(; kwargs...)
    sim_sols_probs = gen_sim_sols_probs(; kwargs...)
    gen_sim_sols(sim_sols_probs)
end

# feed a single poptl NamedTuple
# after calling sim_sols_probs, extract the sol component
function gen_sim_sols(sim_sols_probs)
    let sim_sols_probs = sim_sols_probs
        (poptl; kwargs...) -> begin
            res_sim = sim_sols_probs(poptl.fixed, poptl.random, poptl.indiv,
                poptl.indiv_random;
                kwargs...)
            map(x -> x.sol, res_sim)
        end
    end
end

# function sample_and_add_ranef(problem,
#         priors_random::ComponentVector,
#         rng::AbstractRNG = default_rng();
#         psets)
#     keys_random = keys(priors_random)
#     #keys_gen = (k for k in keys_random) # mappring directly of keys does not work
#     #kp = first(keys(random))
#     tup = map(keys_random) do kp
#         dist_sigma = priors_random[kp]
#         #σ = mean(dist_sigma)
#         σ = rand(rng, dist_sigma)
#         dim_d = length(dist_sigma)
#         len_σ = length(σ)
#         dist = (len_σ == 1) ?
#                fit_mean_Σ(LogNormal, 1, σ) :
#                fit_mean_Σ(MvLogNormal, fill(1, len_σ), PDiagMat(exp.(σ)))
#         rand(rng, dist)
#     end
#     ranef = ComponentVector(; zip(keys_random, tup)...)
#     paropt_r = get_paropt_labeled(psets.random, problem) .* ranef
#     remake(problem, paropt_r, psets.random)
# end

function sample_random(inv_case::AbstractCrossInversionCase, random; scenario,
        rng::AbstractRNG = Random.default_rng())
    priors_random_dict = get_priors_random_dict(inv_case; scenario)
    priors_random = dict_to_cv(keys(random), priors_random_dict)
    random .* sample_ranef(rng, priors_random)
end

function sample_ranef(rng, priors_random)
    gen = (begin
        dist_sigma = priors_random[kp]
        #σ = mean(dist_sigma)
        σ = rand(rng, dist_sigma)
        #dim_d = length(dist_sigma)
        len_σ = length(σ)
        dist = (len_σ == 1) ?
               fit_mean_Σ(LogNormal, 1, σ) :
               fit_mean_Σ(MvLogNormal, fill(1, len_σ), PDiagMat(σ .^ 2))
        r = rand(rng, dist)
        (kp, r)
    end
           for kp in keys(priors_random))
    ComponentVector(; gen...)
end
