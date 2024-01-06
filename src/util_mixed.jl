"""
    setup_psets_fixed_random_indiv(system, keys_popt, keys_opt_fixed, keys_opt_random)

Setup the ProblemParSetters for given system and parameter names.

Only the entries in keys_opt_fixed and keys_opt_random that are actually occurring
in popt_names are set.
The indiv parameters are the difference set between popt_names and the others.

Returns a NamedTuple with `ODEProblemParSetter` for `fixed`, `random`, and `indiv` parameters.
"""
function setup_psets_fixed_random_indiv(system, keys_popt, keys_opt_fixed, keys_opt_random)
    keys_opt_fixed1 = intersect(keys_popt, keys_opt_fixed)
    keys_opt_random1 = intersect(keys_popt, keys_opt_random)
    keys_opt_site1 = setdiff(keys_popt, union(keys_opt_fixed1, keys_opt_random1))
    # psets = (;
    #     fixed = ODEProblemParSetter(system, Axis(keys_opt_fixed1)),
    #     random = ODEProblemParSetter(system, Axis(keys_opt_random1)),
    #     indiv = ODEProblemParSetter(system, Axis(keys_opt_site1)))
    psets = (;
        fixed = ODEProblemParSetter(system, keys_opt_fixed1),
        random = ODEProblemParSetter(system, keys_opt_random1),
        indiv = ODEProblemParSetter(system, keys_opt_site1))
    return psets
end

# function setup_psets_fixed_random_indiv(system, chn::MCMCChains.Chains)
#     @error "not fully implemented yet."
#     keys_opt = extract_popt_names(chn)
#     setup_psets_fixed_random_indiv(system,
#         keys_opt[(:par_names, :keys_opt_fixed, :keys_opt_random)]...)
# end

"""
    sim_sols_probs(fixed, random, indiv, indiv_random; pset_u0, pset_p, u0, p)

Update and simulate system (given with tools to gen_sim_sols_probs) by 
- first update given u0, and p using ProblemParSetter pset_u0 and pset_p
- for each individual i
  - update fixed parameters: fixed 
  - update random parameters: random .* indiv_random[:,i]
  - update site parameters: indiv[:,i]
  - simulate the problem
- return a vector(n_indiv) of (;sol, problem_opt)
"""
function gen_sim_sols_probs(; tools, psets, solver=AutoTsit5(Rodas5()), kwargs_gen...)
    fLogger = EarlyFilteredLogger(current_logger()) do log
        #@show log
        !(log.level == Logging.Warn && log.group == :integrator_interface)
    end
    n_indiv = length(tools)
    # see help on MTKHelpers.ODEProblemParSetterConcrete
    fbarrier = (psets=map(get_concrete, psets),
        problemupdater=get_concrete(first(tools).problemupdater)) -> let solver = solver,
        problems_indiv = map(t -> t.problem, tools),
        psets = psets, problemupdater = problemupdater,
        n_indiv = n_indiv,
        kwargs_gen = kwargs_gen, 
        fLogger = fLogger #, psetci_u_PlantNP=psetci_u_PlantNP 
        #
        (fixed, random, indiv, indiv_random;
            pset_u0=nothing, pset_p=nothing, u0=nothing, p=nothing,
            kwargs...) -> begin
            is_set_u0 = !isnothing(pset_u0) && !isnothing(u0)
            is_set_p = !isnothing(pset_p) && !isnothing(p)
            map(1:n_indiv) do i_indiv
                # need a different tools, because tools.problem.u0 component that are 
                # not optimized may differ
                problem_opt = problems_indiv[i_indiv] # no need to copy, in update_statepar 
                # if u0 or p are provided with corresponding setter, first invoke those
                # u0 and p need to be provided as matrix with one column for each site
                if is_set_u0
                    problem_opt = @inferred remake(problem_opt, u0[:, i_indiv], pset_u0)
                end
                if is_set_p
                    problem_opt = @inferred remake(problem_opt, p[:, i_indiv], pset_p)
                end
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
                    sol = solve(problem_opt, solver; kwargs_gen..., kwargs...)
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

