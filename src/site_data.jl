# """
#     setup_scenario(site, targetlim, scenario, u0, p;
#         system = get_plant_sesam_system(scenario),
#         tspan = (0, 20000),
#         sitedata = get_sitedata(site),    
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
# function setup_popt_scenario(site, targetlim, scenario; kwargs... )
#     u0d, pd = get_initial_sesam_parameters()
#     init_u0_p_poptnames(site, targetlim, scenario, u0d, pd)
# end

# function setup_tools_scenario_u0_popt(site, targetlim, scenario, u0, popt,
#     kwargs...
#     )
#     tools = setup_tools_scenario(site, targetlim, scenario; kwargs...)
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
Variant  of setup_tools_scenario where u0 and p are known,

The order of components in u0 and p may habe changed in system since
states and parameters have been stored.
Hence, update a problem's u0 by `remake(prob, u0=myu0[u_map])`.
"""
function setup_tools_scenario(site, scenario, popt;
        system,
        sitedata = get_sitedata(Val(scenario.system), site, scenario),
        tspan = (0, maximum(map(stream -> maximum(stream.t), sitedata))),
        u0 = nothing,
        p = nothing,)
    sys_num_dict = get_system_symbol_dict(system)
    priors_df = get_priors_df(Val(scenario.system), site, scenario)
    # default u0 and p from expected value of priors
    if isnothing(u0)
        priors_u0 = get_priors(unique(symbol_op.(states(system))), priors_df)
        u0 = meandist2componentarray(priors_u0)
    end
    if isnothing(p)
        priors_p = get_priors(unique(symbol_op.(parameters(system))), priors_df)
        p = meandist2componentarray(priors_p)
    end
    problem = ODEProblem(system, system_num_dict(u0, sys_num_dict), tspan,
        system_num_dict(p, sys_num_dict))
    #
    pset = ODEProblemParSetter(system, popt)
    #
    u_map = get_u_map(keys(u0), pset)
    p_map = get_p_map(keys(p), pset)
    #
    problemupdater = NullProblemUpdater()
    #
    priors_tup = map(keys(popt)) do k
        get_priors(keys(popt[k]), priors_df)
    end
    priors = ComponentVector(; zip(keys(popt),priors_tup)...)
    #
    (; pset, u_map, p_map, problemupdater, priors, problem, sitedata)
end

"""
Take a ComponentArray of possibly multivariate distributions
and return a new ComponentArray of means of each distribution.
"""
meandist2componentarray = function(priors)
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
    get_sitedata(ValueType, site, scenario)

Provide Tuple `(site -> (stream_info)` for each site.
Where StreamInfo is a Tuple `(streamsymbol -> (;t, obs, obs_true))`.
Such that solution can be indexed by sol[streamsymbol](t) to provide
observations.
Value `obs_true` is optional. They are synthetic data without noise 
generated from the system, which are not used in inversion, but used for comparison.

The ValueType dispatches to different implementations. There is 
am implementation for `Val(:CrossInverts_samplesystem1)` independent of scenario.
"""
function get_sitedata end

"""
    get_priors_df(ValueType, site, scenario)

Provide a DataFrame with columns :par, :dist    
"""
function get_priors_df end

"""
    df_from_parmsModeUpperRows(parmsModeUpperRows)

Convert Tuple-Rows of (:par, :dType, :med, :upper) to DataFrame.
And Fit distribution.
"""
function df_from_parmsModeUpperRows(parmsModeUpperRows)
    cols = (:par, :dType, :med, :upper)
    df_dist = rename!(DataFrame(Tables.columntable(parmsModeUpperRows)), collect(cols))
    f1v = (par, dType, med, upper) -> begin
        dist = dist0 = fit(dType, @qp_m(med), @qp_uu(upper))
    end
    DataFrames.transform!(df_dist, Cols(:par, :dType,:med,:upper) => ByRow(f1v) => :dist)
    df_dist
end

"""
    get_priors(pars, priors_df::DataFrame)

Extract the ComponentVector(pars -> Distribution) from priors_df with columns :par and :dist
"""
function get_priors(pars, priors_df::DataFrame)
    @chain priors_df begin
        filter(:par => ∈(pars), _)  # only those rows for pars
        ComponentVector(; zip(_.par, _.dist)...)   
        getindex(_, pars)           # reorder 
    end
end


