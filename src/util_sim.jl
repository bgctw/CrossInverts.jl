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
function gen_sim_sols_probs(; tools, psets, problemupdater, solver = AutoTsit5(Rodas5()), kwargs_gen...)
    fLogger = EarlyFilteredLogger(current_logger()) do log
        #@show log
        !(log.level == Logging.Warn && log.group == :integrator_interface)
    end
    n_indiv = length(tools)
    # see help on MTKHelpers.ODEProblemParSetterConcrete
    fbarrier = () -> let solver = solver,
        problems_indiv = map(t -> t.problem, tools),
        psets = map(get_concrete, psets), problemupdater = get_concrete(problemupdater),
        n_indiv = n_indiv,
        kwargs_gen = kwargs_gen,
        fLogger = fLogger
        #, psetci_u_PlantNP=psetci_u_PlantNP 
        kwargs_indiv_default = fill((), n_indiv)
        #
        (fixed, random, indiv, indiv_random; kwargs_indiv = kwargs_indiv_default, kwargs...) -> begin
            map(1:n_indiv) do i_indiv
                problem_opt = problems_indiv[i_indiv] # no need to copy, in update_statepar 
                # remake Dict not inferred yet: problem_opt = @inferred remake(problem_opt, fixed, psets.fixed)
                problem_opt = remake(problem_opt, fixed, psets.fixed)
                problem_opt = remake(problem_opt,
                    random .* indiv_random[:, i_indiv], psets.random)
                problem_opt = remake(problem_opt, indiv[:, i_indiv], psets.indiv)
                # make sure to use problemupdater on the last update
                #problem_opt = @inferred problemupdater(problem_opt)
                problem_opt = problemupdater(problem_opt)
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
