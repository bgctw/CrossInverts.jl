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
- `get_priors_dict(::AbstractCrossInversionCase, indiv_id; scenario)`
  Priors for model parameters in fixed, random, and indiv effects. 
- `get_priors_random_dict(::AbstractCrossInversionCase; scenario)`
  Priors for meta-parameters for random effects.
- `get_obs_uncertainty_dist_type(::AbstractCrossInversionCase; scenario)`
  Type of distribution of observation-uncertainty per stream.
- `get_indivdata(::AbstractCrossInversionCase, indiv_id; scenario)`
  The times, observations, and uncertainty parameters per indiv_id and stream.
- `get_problemupdater(::AbstractCrossInversionCase; scenario)`
  A ProblemUpdater for ensuring consistent parameters after setting optimized 
  parameters.
"""
abstract type AbstractCrossInversionCase end

"""
    get_problemupdater(::AbstractCrossInversionCase; scenario)

Return a specific `ProblemUpdater` for given Inversioncase and scennario.
It is applied after parameters to optimized have been set. The typical
case is optimizing a single parameter, but adjusting other parameters to
be consistent with the optimized one, e.g. always use the same value for
another parameter.

The default is a `NullProblemUpdater`, which does not modify parameters.    
"""
function get_problemupdater(::AbstractCrossInversionCase; scenario = NTuple{0, Symbol}())
    NullProblemUpdater()
end


"""
    setup_tools_indiv(indiv_id;
            inv_case::AbstractCrossInversionCase, scenario,
            system,
            sitedata = get_indivdata(inv_case, indiv_id; scenario),
            tspan = (0, maximum(map(stream -> stream.t[end], sitedata))),
            u0 = nothing,
            p = nothing,
            keys_indiv = NTuple{0, Symbol}(),
            priors_dict = get_priors_dict(inv_case, indiv_id; scenario),
            u0_default = ComponentVector(),
            p_default = ComponentVector(),
            )

Compiles the information and tools for individuals.
Returns a `NamedTuple`: `(;priors_indiv, problem, sitedata)`.
"""
function setup_tools_indiv(indiv_id;
        inv_case::AbstractCrossInversionCase, scenario,
        system,
        sitedata = get_indivdata(inv_case, indiv_id; scenario),
        tspan = (0, maximum(map(stream -> stream.t[end], sitedata))),
        u0 = nothing,
        p = nothing,
        keys_indiv = NTuple{0, Symbol}(),
        priors_dict = get_priors_dict(inv_case, indiv_id; scenario),
        u0_default = ComponentVector(),
        p_default = ComponentVector(),
        )
    #Main.@infiltrate_main
    sys_num_dict = get_system_symbol_dict(system)
    # default u0 and p from expected value of priors
    u0_vars = unique(symbol_op.(unknowns(system)))
    if isnothing(u0)
        missing_vars = [v for v in u0_vars if v ∉ keys(priors_dict)]
        priors_u0 = dict_to_cv(setdiff(u0_vars, missing_vars), priors_dict)
        u0 = meandist2componentarray(priors_u0)
    end
    p_vars = unique(symbol_op.(parameters(system)))
    if isnothing(p)
        missing_vars = [v for v in p_vars if v ∉ keys(priors_dict)]
        priors_p = dict_to_cv(setdiff(p_vars, missing_vars), priors_dict)
        p = meandist2componentarray(priors_p)
    end
    u0p = ComponentVector(
        state = ComponentVector(u0_default; u0...), # mreging to defaults
        par = ComponentVector(p_default; p...))
    problem = ODEProblem(system, system_num_dict(u0p.state, sys_num_dict), tspan,
        system_num_dict(u0p.par, sys_num_dict))
    #
    priors_indiv = dict_to_cv(keys_indiv, priors_dict)
    #
    (;priors_indiv, problem, sitedata)
end

"""
Put priors for fixed, random, and random_σ into a ComponentVector.
The indiv priors can be indiv_id-specific, they are setup in `setup_tools_indiv`.     
"""
function setup_priors_pop(keys_fixed, keys_random;
        inv_case::AbstractCrossInversionCase, scenario,
        priors_dict = get_priors_dict(inv_case, missing; scenario),
        priors_random_dict = get_priors_random_dict(inv_case; scenario))
    (;
        fixed = dict_to_cv(keys_fixed, priors_dict),
        random = dict_to_cv(keys_random, priors_dict),
        random_σ = dict_to_cv(keys_random, priors_random_dict)
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
    get_obs_uncertainty_dist_type(::AbstractCrossInversionCase; scenario)

Provide the type of distribution of observation uncertainty for given stream,
to be used with `fit_mean_Σ`.
"""
function get_obs_uncertainty_dist_type end


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

Provide a dictionary (par -> Distribution) for prior parameters and unknowns.
"""
function get_priors_dict end


"""
    get_priors_random_dict(::AbstractCrossInversionCase; scenario)

Provide a dictionary (par -> Distribution) for prior parameters and unknowns.
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
    get_indiv_parameters_from_priors(inv_case::AbstractCrossInversionCase,
            indiv_ids, mixed_keys;
            scenario, system,
            rng = StableRNG(234),
            priors_dict = get_priors_dict(inv_case, missing; scenario),
            priors_random_dict = get_priors_random_dict(inv_case; scenario),
            u0_default = ComponentVector(),
            p_default = ComponentVector(),
            )

Construct a DataFrame with parameters across sites with 
- fixed parameters corresponding to the mean of its prior
- random parameters corresponding to mean modified, i.e. multiplied, by sampled 
  random effects and its meta parameters
- individual parameters 

## Value
DataFrame with columns `indiv_id`, `u0` and `p`, with all states and parameters of the
given system as `ComponentVector` labelled by `get_state_labeled` and `get_par_labeled`.

"""
function get_indiv_parameters_from_priors(inv_case::AbstractCrossInversionCase;
        indiv_ids, mixed_keys,
        scenario = NTuple{0, Symbol}(),
        system,
        rng = StableRNG(234),
        priors_dict_indiv = get_priors_dict_indiv(inv_case, indiv_ids; scenario),
        priors_random_dict = get_priors_random_dict(inv_case; scenario),
        u0_default = ComponentVector(),
        p_default = ComponentVector(),
        )
    priors_random = dict_to_cv(mixed_keys.random, priors_random_dict)
    # priors_dict may differ across indiv -> mixed.random and mixed.popt differ
    mixed_indiv = (;
        zip(indiv_ids,
            mean_priors(; mixed_keys, priors_dict, system)
            for
            priors_dict in values(priors_dict_indiv))...)
    map(kc -> check_equal_across_indiv(kc, mixed_indiv), (:fixed, :random))
    psets = setup_psets_mixed(mixed_keys; system, popt = first(mixed_indiv).popt)
    popt_indiv = [label_paropt(psets.popt, mixed.popt) for mixed in mixed_indiv]
    # need to construct problem to properly account for default values
    sdict = get_system_symbol_dict(system)
    problem_indiv = map(popt_indiv) do popt
        u0_numdict = system_num_dict(ComponentVector(u0_default; popt.state...), sdict)        
        p_numdict = system_num_dict(ComponentVector(p_default; popt.par...), sdict)
        ODEProblem(system, u0_numdict, (0.0, 2.0), p_numdict)
    end
    # setup DataFrame and modify u0,p on non-first-row afterwards
    df = DataFrame(indiv_id = collect(indiv_ids), problem = problem_indiv)
    _resample_random = (problem) -> begin
        random = get_paropt_labeled(psets.random, problem)
        r = random .* sample_ranef(rng, priors_random)
        probo = remake(problem, r, psets.random)
        (get_state_labeled(psets.popt, probo), get_par_labeled(psets.popt, probo))
    end
    DataFrames.transform!(df,
        [:problem] => DataFrames.ByRow(_resample_random) => [:u0, :p])
    # in the first row 
    prob1 = df.problem[1]
    # for the first row remove the random effects and stick to the mean
    df[1, [:u0, :p]] .= (get_state_labeled(psets.popt, prob1),
        get_par_labeled(psets.popt, prob1))
    df[:, Not(:problem)]
end

function check_equal_across_indiv(kc, mixed_indiv)
    tmp = ComponentMatrix(hcat((c[kc] for c in mixed_indiv)...),
        first(getaxes(mixed_indiv[1][kc])), FlatAxis())
    rows_cols_equal = map(x -> all(x .== first(x)), eachrow(tmp))
    all(rows_cols_equal) && return ()
    @warn("Expected $kc effects priors to be equal across individuals, "*
    "but means differed in rows $(findall(rows_cols_equal)): $(tmp)")
end

function get_priors_dict_indiv(inv_case, indiv_ids; scenario)
    Dict(id => get_priors_dict(inv_case, id; scenario) for
    id in indiv_ids)
end
