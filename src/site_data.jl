# """
#     setup_scenario(indiv_id, targetlim, scenario, u0, p;
#         system = get_plant_sesam_system(scenario),
#         tspan = (0, 20000),
#         sitedata = get_indivdata(indiv_id),    
#         )

# Initialize the names of parameters to optimize, adjust initial state, 
# setup and ParameterSetters for given scenario

# Returns a NamedTuple with components
# - `u0`: CA.ComponentVector of initial conditions
# - `p`: CA.ComponentVector of parameters, 
# - `pset`: ODEProblemParSetter to update optimized parameters in problem
# - `problemupdater`: ProblemUpdater to adjust other parameter after applying psci, 
# - `problem`: ODEProblem initialized by given u0, and p
# - `system`: ODESystem
# """
# function setup_popt_scenario(indiv_id, targetlim, scenario; kwargs... )
#     u0d, pd = get_initial_sesam_parameters()
#     init_u0_p_poptnames(indiv_id, targetlim, scenario, u0d, pd)
# end

# function setup_tools_scenario_u0_popt(indiv_id, targetlim, scenario, u0, popt,
#     kwargs...
#     )
#     tools = setup_tools_scenario(indiv_id, targetlim, scenario; kwargs...)
#     pset_u0 = ODEProblemParSetter(tools.system, CA.Axis(strip_trailing_zero.(keys(u0))))
#     pset_popt = ODEProblemParSetter(tools.system, CA.Axis(keys(popt)))
#     tools = merge(tools, (;problem=set_u0_popt(
#         tools.problem, u0, popt; pset_u0, pset_popt, problemupdater=tools.problemupdater)))
#     (;tools, pset_u0, pset_popt)
# end

# function set_u0_popt(problem, u0, popt; 
#     pset_u0::AbstractProblemParSetter, 
#     pset_popt::AbstractProblemParSetter, 
#     problemupdater::AbstractProblemUpdater = NullProblemUpdater()
# )
#     problem = remake(problem, u0, pset_u0)
#     problem = remake(problem, popt, pset_popt)
#     problem = problemupdater(problem)
# end

"""
    AbstractCrossInversionCase

Interface for providing all relevant information for a cross-individual
mixed effects bayesian inversion.

Concrete types should implement
- `get_priors_dict(::AbstractCrossInversionCase, indiv_id; scenario)`
  Priors for model parameters in fixed, random, and indiv effects. 
- `get_priors_random_dict(::AbstractCrossInversionCase; scenario)`
  Priors for meta-parameters for random effects.
- `get_obs_uncertainty_dist_type(::AbstractCrossInversionCase; scenario)`
  Type of distribution of observation-uncertainty per stream.
- `get_indivdata(::AbstractCrossInversionCase, indiv_id; scenario)`
  The times, observations, and uncertainty parameters per indiv_id and stream.
"""
abstract type AbstractCrossInversionCase end

function mean_priors(; system, priors_dict, component_keys...)
    #(_comp,_keys) = first(pairs(component_keys))
    gen = (begin
        priors_k = dict_to_cv(_keys, priors_dict)
        m = meandist2componentarray(priors_k)
    end
           for (_comp, _keys) in pairs(component_keys))
    ntup = (; zip(keys(component_keys), gen)...)
    popt = vcat_statesfirst(ntup...; system)
    (; ntup..., popt)
end

"""
TODO describe
"""
function setup_tools_scenario(indiv_id;
        inv_case::AbstractCrossInversionCase, scenario,
        system,
        sitedata = get_indivdata(inv_case, indiv_id; scenario),
        tspan = (0, maximum(map(stream -> stream.t[end], sitedata))),
        u0 = nothing,
        p = nothing,
        keys_indiv = NTuple{0, Symbol}(),)
    #Main.@infiltrate_main
    sys_num_dict = get_system_symbol_dict(system)
    priors_dict = get_priors_dict(inv_case, indiv_id; scenario)
    # default u0 and p from expected value of priors
    if isnothing(u0)
        priors_u0 = dict_to_cv(unique(symbol_op.(states(system))), priors_dict)
        u0 = meandist2componentarray(priors_u0)
    end
    if isnothing(p)
        priors_p = dict_to_cv(unique(symbol_op.(parameters(system))), priors_dict)
        p = meandist2componentarray(priors_p)
    end
    u0p = ComponentVector(state = u0, par = p)
    problem = ODEProblem(system, system_num_dict(u0, sys_num_dict), tspan,
        system_num_dict(p, sys_num_dict))
    #
    pset_u0p = ODEProblemParSetter(system, u0p)
    problem = remake(problem, u0p, pset_u0p)
    #
    # u_map = get_u_map(keys(u0), pset_u0p)
    # p_map = get_p_map(keys(p), pset_u0p)
    #
    problemupdater = NullProblemUpdater()
    #
    #popt_l = label_paropt(pset_u0p, u0p) # axis with split state and par
    #popt_flat = flatten1(popt_l)
    priors_indiv = dict_to_cv(keys_indiv, priors_dict)
    #
    (; pset_u0p, problemupdater, priors_indiv, problem, sitedata)
end

function setup_priors_pop(keys_fixed, keys_random;
        inv_case::AbstractCrossInversionCase, scenario)
    priors_dict = get_priors_dict(inv_case, :unknown_site; scenario)
    priors_random_dict = get_priors_random_dict(inv_case; scenario)
    (;
        fixed = dict_to_cv(keys_fixed, priors_dict),
        random = dict_to_cv(keys_random, priors_dict),
        random_σ = dict_to_cv(keys_random, priors_random_dict),
        # the indiv priors can be indiv_id-specific, they are setup in setup_tools_scenario
    )
end

"""
Take a ComponentArray of possibly multivariate distributions
and return a new ComponentArray of means of each distribution.
"""
meandist2componentarray = function (priors)
    # need to first create several ComponentVectors and then reduce
    # otherwise map on mixing Scalars and Vectors yields eltype any
    @chain priors begin
        map(keys(_)) do k
            ComponentVector(NamedTuple{(k,)}(Ref(mean(_[k]))))
        end
        reduce(vcat, _)
    end
end

"""
    get_indivdata(::AbstractCrossInversionCase, indiv_id; scenario)

Provide Tuple `(indiv_id -> (stream_info)` for each indiv_id.
Where StreamInfo is a Tuple `(streamsymbol -> (;t, obs, obs_true))`.
Such that solution can be indexed by sol[streamsymbol](t) to provide
observations.
Value `obs_true` is optional. They are synthetic data without noise 
generated from the system, which are not used in inversion, but used for comparison.

The ValueType dispatches to different implementations. There is 
am implementation for `Val(:CrossInverts_samplesystem1)` independent of scenario.
"""
function get_indivdata end

"""
    get_priors_dict(::AbstractCrossInversionCase, indiv_id; scenario)

Provide a dictionary (par -> Distribution) for prior parameters and states.
"""
function get_priors_dict end

"""
    get_priors_random_dict(::AbstractCrossInversionCase; scenario)

Provide a dictionary (par -> Distribution) for prior parameters and states.
"""
function get_priors_random_dict end

"""
    df_from_paramsModeUpperRows(paramsModeUpperRows)

Convert Tuple-Rows of (:par, :dType, :med, :upper) to DataFrame.
And Fit distribution.
"""
function df_from_paramsModeUpperRows(paramsModeUpperRows)
    cols = (:par, :dType, :med, :upper)
    df_dist = rename!(DataFrame(Tables.columntable(paramsModeUpperRows)), collect(cols))
    f1v = (par, dType, med, upper) -> begin
        dist = dist0 = fit(dType, @qp_m(med), @qp_uu(upper))
    end
    DataFrames.transform!(df_dist, Cols(:par, :dType, :med, :upper) => ByRow(f1v) => :dist)
    df_dist
end

"""
    dict_to_cv(keys, dict::DataFrame)

Extract the ComponentVector(keys -> Distribution) from dict(:par => :dist)
"""
function dict_to_cv(keys, dict::AbstractDict)
    isempty(keys) ?
    ComponentVector{eltype(values(dict))}() :
    ComponentVector(; zip(keys, getindex.(Ref(dict), keys))...)
    # @chain dict begin
    #     filter(:par => ∈(keys), _)  # only those rows for keys
    #     ComponentVector(; zip(_.par, _.dist)...)
    #     getindex(_, keys)           # reorder 
    # end
end

# """
#     extract_stream_obsmatrices(;tools, vars=(:obs,))

# Extract matrices by stream from sitedata in tools.

# For each indiv_id, tools holds the observations of all streams in property sitedata.
# This function returns a ComponentVector for each stream.
# For each stream it reports subvections t, and matrix for vars where each column
# relates to one indiv_id.
# It checks that all sites report the same time, and that variables have the same
# length as the time column.
# """
# function extract_stream_obsmatrices(; tools, vars = (:obs,))
#     obs = map(t -> t.sitedata, tools)
#     stream_names = keys(first(obs))
#     tup = map(stream_names) do sk
#         #sk = stream_names[1]
#         stream_sites = (obs_site[sk] for obs_site in obs)
#         #collect(stream_sites)
#         #ss = first(stream_sites)
#         ts = [ss.t for ss in stream_sites]
#         all(ts[2:end] .== Ref(ts[1])) ||
#             error("Expected equal time for stream $sk across individuals, but got $ts")
#         nt = length(ts[1])
#         tup_vars = map(vars) do var
#             _data = hcat((ss[var] for ss in stream_sites)...)
#             size(_data, 1) == nt ||
#                 error("Expected variable $var in stream $sk to be of length(time)=$nt) " *
#                       "but was $(size(_data,1))")
#             _data
#         end
#         ComponentVector(; t = ts[1], zip(vars, tup_vars)...)
#     end
#     ComponentVector(; zip(stream_names, tup)...)
# end

"""
    get_obs_uncertainty_dist_type(::AbstractCrossInversionCase; scenario)

Provide the type of distribution of observation uncertainty for given stream,
to be used with `fit_mean_Σ`.
"""
function get_obs_uncertainty_dist_type end
