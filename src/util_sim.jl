# codecov wrongly reports line in fLogger in closure, that actually is applied during solution

"""
    gen_sim_sols_probs(; tools, psets, problemupdater, 
        solver = AutoTsit5(Rodas5()), kwargs_gen...)

Generates a function `sim_sols_probs(fixed, ranadd, ranmul, indiv, indiv_ranmul, indiv_ranadd)`
that updates and simulate the system (given with tools to `gen_sim_sols_probs`) by 
- for each individual i
  - update fixed parameters: fixed 
  - update ranadd parameters: ranadd .+ indiv_ranadd[:,i]
  - update ranmul parameters: ranmul .* indiv_ranmul[:,i]
  - update indiv_id parameters: indiv[:,i]
  - simulate the problem
- return a vector(n_indiv) of (;sol, problem_opt)

If non-optimized `p` or `u0` differ between individuals, they must already be
set in `tools[i_indiv].problem`.
"""
function gen_sim_sols_probs(;
        tools, psets, problemupdater, solver = AutoTsit5(Rodas5()), kwargs_gen...)
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
        (fixed, ranadd, ranmul, indiv, indiv_ranadd, indiv_ranmul;
        kwargs_indiv = kwargs_indiv_default, kwargs...) -> begin
            map(1:n_indiv) do i_indiv
                problem_opt = problems_indiv[i_indiv] # no need to copy, in update_statepar 
                # remake Dict not inferred yet: problem_opt = @inferred remake(problem_opt, fixed, psets.fixed)
                problem_opt = remake(problem_opt, fixed, psets.fixed)
                problem_opt = remake(problem_opt,
                    ranadd .+ indiv_ranadd[:, i_indiv], psets.ranadd)
                problem_opt = remake(problem_opt,
                    ranmul .* indiv_ranmul[:, i_indiv], psets.ranmul)
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

function gen_sim_sols(ax, n_indiv; kwargs...)
    sim_sols_probs = gen_sim_sols_probs(; kwargs...)
    gen_sim_sols(sim_sols_probs, ax, n_indiv)
end

# feed a single poptl NamedTuple
# after calling sim_sols_probs, extract the sol component
# provide a template ComponentVector, such as pop_info.effect_pos
function gen_sim_sols(sim_sols_probs, ax_template::CA.ComponentVector, n_indiv)
    gen_sim_sols(sim_sols_probs, first(CA.getaxes(ax_template)), n_indiv)
end
function gen_sim_sols(sim_sols_probs, ax::CA.Axis, n_indiv)
    let sim_sols_probs = sim_sols_probs, ax=ax, n_indiv=n_indiv
        "simulate for given sample x and retun solution for each individual."
        (x::Union{AbstractVector,NamedTuple}; kwargs...) -> begin
            # attach axis if its not yet a ComponentVectr
            xl = x isa Union{CA.ComponentVector,NamedTuple} ? x : CA.ComponentVector(x, ax)
            (; fixed, ranadd, ranmul, indiv, indiv_ranadd, indiv_ranmul) = xl
            # assume all parameters of individual together -> ncol indivs
            indiv_m = reshape(indiv, :, n_indiv)
            indiv_ranadd_m = reshape(indiv_ranadd, :, n_indiv)
            indiv_ranmul_m = reshape(indiv_ranmul, :, n_indiv)
            res_sim = sim_sols_probs(fixed, ranadd, ranmul, indiv_m,
                indiv_ranadd_m, indiv_ranmul_m;
                kwargs...)
            map(x -> x.sol, res_sim)
        end
    end
end

function sample_random_σ(rng, effect_keys, priors_random_dict)
    # use generator instead of map, because effect_keys is a NamedTuple
    _gen = ((kp, rand(rng, priors_random_dict[kp])) for kp in effect_keys)
    CA.ComponentVector(; _gen...)
end

function get_ranadd_dist(ranadd_σ)
    map(ranadd_σ) do σ
        len_σ = length(σ)
        dist = (len_σ == 1) ?
               fit_mean_Σ(Normal, 0, σ) :
               fit_mean_Σ(Normal, fill(0, len_σ), PDiagMat(σ .^ 2))
    end
end

function get_ranmul_dist(ranmul_σ)
    map(ranmul_σ) do σ
        len_σ = length(σ)
        dist = (len_σ == 1) ?
               fit_mean_Σ(LogNormal, 1, σ) :
               fit_mean_Σ(MvLogNormal, fill(1, len_σ), PDiagMat(σ .^ 2))
    end
end

# Lean from failure: defined function rand in module CrossInverts
# This caused calls to rand (from )
# """
#     rand(rng::Random.AbstractRNG, dist_cp::ComponentVector{<:Distribution})

# Sample a value from each Distribution in dist_cp.    
# """
# function rand(rng::Random.AbstractRNG, dist_cp::ComponentVector{<:Distribution})
#     gen = ((kp, rand(rng, dist_cp[kp])) for kp in keys(dist_cp))
#     ComponentVector(; gen...)
# end
