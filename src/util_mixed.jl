"""
    gen_compute_indiv_rand(pset, random)

Generate a function closure `compute_indiv_rand(u0, p)` that for each population
random effect-mean `random` computes the individual random effect.
It uses the provided `ProblemParSetter` to extract the optimized parameters
from u0 and p.
It is used to get an initial estimate of the random effects given a population
mean, and the individual site parameters.
"""
function gen_compute_indiv_rand(pset::AbstractProblemParSetter, random) 
    let pset=pset, random=random
        compute_indiv_rand = (u0, p) -> begin
            local popt = get_paropt_labeled(pset, u0, p; flatten1=Val(true))
            # k = first(keys(random))
            gen = (popt[k] ./ random[k] for k in keys(random))
            v = reduce(vcat, gen)
            MTKHelpers.attach_axis(v, first(getaxes(random)))
        end
    end # let
end

"""
    setup_psets_fixed_random_indiv(system, popt, fixed, random)

Setup the ProblemParSetters for given system and parameter names.
Assume, that parameters are fiven in flat format, i.e. not state and par labels.
Make sure, that popt holds state entries first.

Only the entries in fixed and random that are actually occurring
in popt_names are set.
The indiv parameters are the difference set between popt_names and the others.

Returns a NamedTuple with `ODEProblemParSetter` for `fixed`, `random`, and `indiv` parameters.
"""
function setup_psets_fixed_random_indiv(system, popt, keys_fixed, keys_random)
    fixed1 = intersect(keys(popt), keys_fixed)
    random1 = intersect(keys(popt), keys_random)
    indiv1 = setdiff(keys(popt), union(fixed1, random1))
    psets = (;
        fixed = ODEProblemParSetter(system, popt[fixed1]),
        random = ODEProblemParSetter(system, popt[random1]),
        indiv = ODEProblemParSetter(system, popt[indiv1]),)
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
  - update site parameters: indiv[:,i]
  - simulate the problem
- return a vector(n_indiv) of (;sol, problem_opt)

If non-optimized p and u0 differ between individuals, they must already be
set in tools[i_indiv].problem.
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
        (fixed, random, indiv, indiv_random; kwargs...) -> begin
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

function sample_and_add_ranef(problem, priors_random::ComponentVector, rng::AbstractRNG=default_rng(); psets) 
    keys_random = keys(priors_random)
    #keys_gen = (k for k in keys_random) # mappring directly of keys does not work
    #kp = first(keys(random))
    tup = map(keys_random) do kp
        dist_sigma = priors_random[kp]
        #sigma_star_d = mean(dist_sigma)
        sigma_star_d = rand(rng, dist_sigma)
        dim_d = length(dist_sigma)
        # TODO instead of sampling independent fit Multivariate LogNormal with marginal
        # expectations of 1
        dist_scalar = map(sigma_star_d) do sigma_star
            fit(LogNormal, 1, Σstar(sigma_star)) # Σstar
        end
        dist = dim_d == 1 ? dist_scalar : product_distribution(dist_scalar...)
        rand(rng, dist)
    end
    ranef = ComponentVector(;zip(keys_random, tup)...)
    paropt_r = get_paropt_labeled(psets.random, problem) .* ranef
    remake(problem, paropt_r, psets.random)
end
