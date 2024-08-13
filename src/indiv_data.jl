"""
    AbstractCrossInversionCase

Interface for providing all relevant information for a cross-individual
mixed effects bayesian inversion.

Concrete types should implement
- `get_case_inverted_system(::AbstractCrossInversionCase; scenario)` 
- `get_case_mixed_keys(::AbstractCrossInversionCase; scenario)`
- `get_case_indiv_ids(::AbstractCrossInversionCase; scenario)`
- `get_case_priors_dict(::AbstractCrossInversionCase, indiv_id; scenario)`
  Priors for optimized model effects, which may differ by individual. 
- `get_case_priors_random_dict(::AbstractCrossInversionCase; scenario)`
  Priors for meta-parameters for ranadd and ranmul effects.
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

Provide NamedTuple `(;fixed, ranadd, ranmul, indiv)` of tuples of parameter names (Symbol) that
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
    get_case_priors_random_dict(::AbstractCrossInversionCase; scenario)

Provide a dictionary (par -> Distribution) for hyperpriors
of the spread of the ranmul and ranadd effects.
"""
function get_case_priors_random_dict end

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


"""
Generate observations and uncertainties according to priors

Uncertainty parameters of different streams are given with Dictionary 
`unc_par`. 
The type of distribution is obtained from 
`get_case_obs_uncertainty_dist_type(inv_case, s; scenario)`.

For Normal and LogNormal this is the σ standard deviation parameter.
For MvNormal and MvLogNormal this is the Σ Covariance matrix.
Typical LogNormal is `convert(Matrix, PDiagMat(log.([1.1, 1.1])))`.

"""
function simulate_indivdata(;inv_case, scenario, unc_par, solver = Tsit5(), rng = StableRNG(123))
    # using and setup in test_util_mixed
    system_u0_p_default = get_case_inverted_system(inv_case; scenario)
    (;system, u0_default, p_default) = system_u0_p_default
    defaults(system)
    #indiv_ids = get_case_indiv_ids(inv_case; scenario)
    (; p_indiv, ranadd_dist_cv, ranmul_dist_cv) = get_indiv_parameters_from_priors(
        inv_case; scenario, system_u0_p_default, rng)
    # p_indiv = get_indiv_parameters_from_priors(inv_case; scenario, indiv_ids, mixed_keys,
    #     system_u0_p_default = (; system,
    #         u0_default = CA.ComponentVector(), p_default = CA.ComponentVector(sv₊i2 = 0.1)))
    #using DistributionFits, StableRNGs, Statistics
    # other usings from test_util_mixed
    _dict_nums = get_system_symbol_dict(system)
    # setup a problem, numbers do not matter, because set below from p_indiv
    t = [0.0]
    problem = ODEProblem(system, system_num_dict(u0_default, _dict_nums), (0.0, maximum(t)), system_num_dict(p_default, _dict_nums))
    #indiv_id = first(keys(p_indiv))
    streams = collect(keys(unc_par))
    dtypes = Dict(s => get_case_obs_uncertainty_dist_type(inv_case, s; scenario)
    for s in streams)
    d_noise = Dict(s => begin
                       unc = unc_par[s]
                       m0 = ((dtypes[s] <: Normal) || (dtypes[s] <: MvNormal)) ? 0.0 : 1.0
                       m = unc isa AbstractMatrix ? fill(m0, size(unc, 1)) : m0
                       fit_mean_Σ(dtypes[s], m, unc)
                   end for s in streams)
    # d_noise[:sv₊x]
    indiv_dict = Dict(p_indiv.indiv_id .=> zip(p_indiv.u0, p_indiv.p))
    # indiv_id = first(p_indiv.indiv_id)
    obs_tuple = map(p_indiv.indiv_id) do indiv_id
        #st = Dict(Symbolics.scalarize(sv.x .=> p_indiv[indiv_id].u0.sv₊x))
        #p_new = Dict(sv.i .=> p_indiv[indiv_id].sv₊i)
        #prob = ODEProblem(system, st, (0.0, 2.0), p_new)
        u0_dict = system_num_dict(indiv_dict[indiv_id][1], _dict_nums)
        p_dict = system_num_dict(indiv_dict[indiv_id][2], _dict_nums)
        probo = remake(problem, u0 = u0_dict, p = p_dict)
        #pset = ODEProblemParSetter(get_system(probo), Symbol[])        
        #x1 = get_state_labeled(pset, probo).x1
        #x2 = get_state_labeled(pset, probo).x2
        #get_par_labeled(pset, probo)
        sol = solve(probo, solver, saveat = t)
        #sol[[sv.x[1], sv.dec2]]
        #sol[_dict_nums[:sv₊dec2]]
        #stream = last(collect(streams)) #stream = first(streams)
        tmp = map(streams) do stream
            obs_true = sol[Symbolics.scalarize(_dict_nums[stream])]
            n_obs = length(obs_true)
            obs_unc = fill(unc_par[stream], n_obs)  # may be different for each obs
            noise = rand(rng, d_noise[stream], n_obs)
            obs = if dtypes[stream] <: Union{Normal, MvNormal}
                length(size(noise)) == 1 ? 
                obs = obs_true .+ noise :
                obs = map(obs_true, eachcol(noise)) do obs_i, noise_i
                    obs_i .+ noise_i # within each timepoint add generated noise
                end
            else
                length(size(noise)) == 1 ?
                obs = obs_true .* noise :
                obs = map(obs_true, eachcol(noise)) do obs_i, noise_i
                    obs_i .* noise_i
                end
            end
            (; t, obs, obs_unc, obs_true)
        end
        (; zip(streams, tmp)...)
    end
    res = (;indivdata=(; zip(p_indiv.indiv_id, obs_tuple)...), 
        p_indiv, d_noise, ranadd_dist_cv, ranmul_dist_cv)
end

