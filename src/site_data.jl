# """
#     setup_scenario(indiv_id, targetlim, scenario, u0, p;
#         system = get_plant_sesam_system(scenario),
#         tspan = (0, 20000),
#         indivdata = get_case_indivdata(indiv_id),    
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
#     tools = setup_tools_indiv(indiv_id, targetlim, scenario; kwargs...)
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
- `get_case_inverted_system(::AbstractCrossInversionCase; scenario)` 
- `get_case_mixed_keys(::AbstractCrossInversionCase; scenario)`
- `get_case_indiv_ids(::AbstractCrossInversionCase; scenario)`
- `get_case_priors_dict(::AbstractCrossInversionCase, indiv_id; scenario)`
  Priors for model parameters in fixed, random, and indiv effects. 
- `get_case_riors_random_dict(::AbstractCrossInversionCase; scenario)`
  Priors for meta-parameters for random effects.
- `get_case_obs_uncertainty_dist_type(::AbstractCrossInversionCase; scenario)`
  Type of distribution of observation-uncertainty per stream.
- `get_case_indivdata(::AbstractCrossInversionCase, indiv_id; scenario)`
  The times, observations, and uncertainty parameters per indiv_id and stream.
optionally:
- `get_case_u0p(::AbstractCrossInversionCase; scenario)`  
- `get_case_problemupdater(::AbstractCrossInversionCase; system, scenario)`
  A ProblemUpdater for ensuring consistent parameters after setting optimized 
  parameters.
"""
abstract type AbstractCrossInversionCase end

"""
    get_case_problemupdater(::AbstractCrossInversionCase; scenario)

Return a specific `ProblemUpdater` for given Inversioncase and scennario.
It is applied after parameters to optimized have been set. The typical
case is optimizing a parameter, but adjusting other non-optimized parameters to
be consistent with the optimized one, e.g. always use the same value for
another parameter.

The default is a `NullProblemUpdater`, which does not modify parameters.    
"""
function get_case_problemupdater(::AbstractCrossInversionCase; 
    system, scenario = NTuple{0, Symbol}())
    NullProblemUpdater()
end

"""
    get_case_u0p(::AbstractCrossInversionCase; scenario = NTuple{0, Symbol}())

Return initial values and parameters to set in Problem of each individual.
If the Data.Frame holds a row for a specific indiv_id, the provided values 
override the initial estimate from the priors. 
This can be used to start optimization from a given state.

The default has no information:
`DataFrame(indiv_id = Symbol[], u0 = ComponentVector[], p = ComponentVector[])`.
"""
function get_case_u0p(::AbstractCrossInversionCase; scenario = NTuple{0, Symbol}())
    DataFrame(indiv_id = Symbol[], u0 = ComponentVector[], p = ComponentVector[])
end

"""
    get_case_inverted_system(::AbstractCrossInversionCase; scenario)

Provide a `NamedTuple` 
`(;system::AbstractSystem, u0_default::ComponentVector, p_default::ComponentVector)`
of the inverted system and default initial values and parameters 
for given inversion scenario.
"""
function get_case_inverted_system end
"""
    get_case_mixed_keys(::AbstractCrossInversionCase; scenario)

Provide NamedTuple `(;fixed, random, indiv)` of tuples of parameter names (Symbol) that
are optimized in the inversion scenario.
"""
function get_case_mixed_keys end

"""
    get_case_indiv_ids(::AbstractCrossInversionCase; scenario)

Provide Tuple of Symbols identifying the individuals.
"""
function get_case_indiv_ids end


"""
    get_case_obs_uncertainty_dist_type(::AbstractCrossInversionCase; scenario)

Provide the type of distribution of observation uncertainty for given stream,
to be used with `fit_mean_Σ`.
"""
function get_case_obs_uncertainty_dist_type end


"""
    get_case_indivdata(::AbstractCrossInversionCase, indiv_id; scenario)

Provide Tuple `(indiv_id -> (stream_info)` for each indiv_id.
Where StreamInfo is a Tuple `(streamsymbol -> (;t, obs, obs_true))`.
Such that solution can be indexed by sol[streamsymbol](t) to provide
observations.
Value `obs_true` is optional. They are synthetic data without noise 
generated from the system, which are not used in inversion, but used for comparison.

The ValueType dispatches to different implementations. There is 
am implementation for `Val(:CrossInverts_samplesystem1)` independent of scenario.
"""
function get_case_indivdata end

"""
    get_case_priors_dict(::AbstractCrossInversionCase, indiv_id; scenario)

Provide a dictionary (par -> Distribution) for prior parameters and unknowns.
"""
function get_case_priors_dict end


"""
    get_case_riors_random_dict(::AbstractCrossInversionCase; scenario)

Provide a dictionary (par -> Distribution) for prior parameters and unknowns.
"""
function get_case_riors_random_dict end

"""
    df_from_paramsModeUpperRows(paramsModeUpperRows)

Convert Tuple-Rows of `(:par, :dType, :mode, :upper)` to `DataFrame`.
And fit distribution and report it in column `:dist`.
"""
function df_from_paramsModeUpperRows(paramsModeUpperRows)
    cols = (:par, :dType, :mode, :upper)
    df_dist = rename!(DataFrame(Tables.columntable(paramsModeUpperRows)), collect(cols))
    f1v = (par, dType, mode, upper) -> begin
        dist = dist0 = fit(dType, mode, @qp_uu(upper), Val(:mode))
    end
    DataFrames.transform!(df_dist, Cols(:par, :dType, :mode, :upper) => ByRow(f1v) => :dist)
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

# Extract matrices by stream from indivdata in tools.

# For each indiv_id, tools holds the observations of all streams in property indivdata.
# This function returns a ComponentVector for each stream.
# For each stream it reports subvections t, and matrix for vars where each column
# relates to one indiv_id.
# It checks that all sites report the same time, and that variables have the same
# length as the time column.
# """
# function extract_stream_obsmatrices(; tools, vars = (:obs,))
#     obs = map(t -> t.indivdata, tools)
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


