"""
    setup_inversion(inv_case::AbstractCrossInversionCase; scenario = NTuple{0, Symbol}())

Calls all the functions for specific [`AbstractCrossInversionCase`](@ref) to setup the
inversion.

Returns a `NamedTuple` with entries: `(; system, pop_info, indiv_info)`

`pop_info` is a NamedTuple with entriees
- `mixed_keys`: The optimized parameters and their distribution across individual 
  as returned by [`get_case_mixed_keys]`(@ref)
- `indiv_ids`: A tuple of ids (Symbols) of the individuals taking part in the 
   inversion, as returned by [`get_case_indiv_ids]`(@ref)
- `mixed`: mixed effects `NamedTuple(fixed, ranadd, ranmul, indiv, indiv_ranadd, indiv_ranmul)`
  from individual's states and parameters in the format expected by forward sim.
- `psets`: `NTuple{ODEProblemParSetter}` for each mixed component
- `priors_pop`: `ComponentVector` of priors on population level 
  `(fixed, ranadd, ranmul, ranadd_σ, ranmul_σ)`
- `problemupdaterproblemupdater`: used to update problems after setting parameters
  to be optimized.
- `sample0`: ComponentVector of an initial sample
  This can be used to name (attach_axis) a sample from MCMCChains object 
- `effect_pos`: ComponentVector to map keys to positions (1:n_par) in sample

`indiv_info`: is a DataFrame with rows for each individual with  columns
  - `indiv_id`: Symbols identifying individuals
  - `u0` and `p`: ComponentVectors of initial states and parameters
  - `paropt`: optimized parameters extracted from indiviudals state and parameters
  - `tools`: tools initialized for each site 
    - `priors_indiv`: priors, which may differ between individuals
    - `problem`: the ODEProblem set up with initial state and parameters
    - `indivdata`: as returned by get_case_indivdata
"""
function setup_inversion(inv_case::AbstractCrossInversionCase;
        scenario = NTuple{0, Symbol}(),
        system_u0_p_default = get_case_inverted_system(inv_case; scenario)
)
    #    pop_info = (;mixed, psets, problemupdater, priors_pop, sample0, effect_pos)
    (; system, u0_default, p_default) = system_u0_p_default
    mixed_keys = get_case_mixed_keys(inv_case; scenario)
    indiv_ids = get_case_indiv_ids(inv_case; scenario)
    priors_dicts = get_priors_dict_indiv(inv_case, indiv_ids; scenario)
    psets = setup_psets_mixed(mixed_keys; system = system_u0_p_default.system)
    #
    indivdata = [get_case_indivdata(inv_case, id; scenario) for id in indiv_ids]
    tspans = map(indivdata) do indivdata_i
        (0, maximum(map(stream -> stream.t[end], indivdata_i)))
    end
    problems = setup_indiv_problems(; inv_case, scenario, tspans, priors_dicts)
    #id = first(indiv_ids)
    #problem = first(problems)
    u0 = map(prob -> get_state_labeled(psets.popt, prob), problems)
    p = map(prob -> get_par_labeled(psets.popt, prob), problems)
    paropt = map(prob -> get_paropt_labeled(psets.popt, prob), problems)
    priors_indiv = [dict_to_cv(mixed_keys.indiv, priors_dicts[id]) for id in indiv_ids]
    tools = map(priors_indiv, problems, indivdata) do priors_indiv, problem, indivdata
        (; priors_indiv, problem, indivdata)
    end
    indiv_info = DataFrame(indiv_id = collect(indiv_ids); u0, p, paropt, tools)
    # (; pop_info, indiv_info) = setup_tools_mixed(indiv_info; 
    #     inv_case, scenario, system_u0_p_default, mixed_keys)
    pop_info = setup_tools_mixed(indiv_info; inv_case, scenario, system, psets)
    pop_info = (; mixed_keys, indiv_ids, system_u0_p_default, pop_info...)
    (; system, pop_info, indiv_info)
end

function setup_tools_mixed(indiv_info::DataFrame;
        inv_case, scenario, system, psets
)
    mixed = extract_mixed_effects(psets, indiv_info.paropt)
    priors_pop = setup_priors_pop(
        keys(mixed.fixed), keys(mixed.ranadd), keys(mixed.ranmul); inv_case, scenario)
    #vcat returns a plain vector, if types are different
    priors_σ = ComponentVector(;
        zip(
            vcat(keys(priors_pop.ranadd_σ)..., keys(priors_pop.ranmul_σ)...),
            vcat(priors_pop.ranadd_σ, priors_pop.ranmul_σ)
        )...)
    sample0 = get_init_mixedmodel(psets,
        indiv_info.paropt, priors_σ;
        indiv_ids = indiv_info.indiv_id)
    effect_pos = MTKHelpers.attach_axis(1:length(sample0), MTKHelpers._get_axis(sample0))
    problemupdater = get_case_problemupdater(inv_case; system, scenario)
    #
    pop_info = (; mixed, psets, problemupdater, priors_pop, sample0, effect_pos)
end

"""
    setup_psets_mixed(mixed_keys; system)
    
Creates the `ODEProblemUpdaters` given several optimized parameters and a system.
Creates a separate Updater for each class of optimized parameters in mixed_keys,
and another one for the union of all parameters.
"""
function setup_psets_mixed(mixed_keys; system)
    gen = ((kc, ODEProblemParSetter(system, mixed_keys[kc])) for
    kc in keys(mixed_keys))
    popt_keys = reduce((c, x) -> (c..., x...), mixed_keys)
    pset_popt = LoggingExtras.withlevel(Logging.Error) do
        # warning on states not ordered first - ok here
        ODEProblemParSetter(system, popt_keys)
    end
    psets = (; gen..., popt = pset_popt)
    return psets
end

"""
    mean_priors(; mixed_keys, priors_dict, system)

Compute the means of prior distributions for the parameters listed in
components of mixed_keys and add another popt component that aggregates
all the components, but states first.

## Arguments
- `mixed_keys`: NamedTuple with entry `NTuple{Symbol}` for each class of 
  optimized parameters
- `priors_dict`: Dictionary of prior distribution for parameters
- `system`: The AbstractSystem used to distinguish states and parameters 

Returns a NamedTuple with same entries as mixed_keys holding ComponentVectors     
of means of the parameter distributions.
"""
function mean_priors(; mixed_keys, priors_dict, system)
    #(_comp,_keys) = first(pairs(mixed_keys))
    gen = (
        begin
            priors_k = dict_to_cv(_keys, priors_dict)
            meandist2componentarray(priors_k)
        end
    for (_comp, _keys) in pairs(mixed_keys)
    )
    ntup = (; zip(keys(mixed_keys), gen)...)
    popt = vcat_statesfirst(ntup...; system)
    (; ntup..., popt)
end

"""
Create problem for each individual in indiv_ids. 
Initial states and parameters are merged, i.e. overridden in the following order
- default in the system
- default provided with `system_u0_p_default`
- mean of prior
- value provided with indiv_u0p
"""
function setup_indiv_problems(;
        inv_case, scenario, tspans,
        system_u0_p_default = get_case_inverted_system(inv_case; scenario),
        indiv_ids = get_case_indiv_ids(inv_case; scenario),
        indiv_u0p = get_case_u0p(inv_case; scenario),
        priors_dicts = get_priors_dict_indiv(inv_case, indiv_ids; scenario)
)
    (; system, u0_default, p_default) = system_u0_p_default
    # complete the u0p specification by adding empty ComponentVectors for missing indivs
    df_spec = subset(indiv_u0p, :indiv_id => ByRow(∈(indiv_ids)))
    df = leftjoin(DataFrame(indiv_id = collect(indiv_ids)), df_spec, on = :indiv_id)
    DataFrames.transform!(
        df, [:u0, :p] .=> ByRow(x -> coalesce(x, ComponentVector())) .=> [:u0, :p])
    DataFrames.disallowmissing!(df)
    sys_num_dict = get_system_symbol_dict(system)
    u0_vars = unique(symbol_op.(unknowns(system)))
    p_vars = unique(symbol_op.(parameters(system)))
    #indiv_id = first(indiv_ids); priors_dict = first(priors_dicts)
    problems = map(indiv_ids, df.u0, df.p, tspans) do indiv_id, u0, p, tspan
        priors_dict = priors_dicts[indiv_id]
        missing_vars = [v for v in u0_vars if v ∉ keys(priors_dict)]
        priors_dict_u0 = dict_to_cv(setdiff(u0_vars, missing_vars), priors_dict)
        u0_priors = meandist2componentarray(priors_dict_u0)
        missing_vars = [v for v in p_vars if v ∉ keys(priors_dict)]
        priors_dict_p = dict_to_cv(setdiff(p_vars, missing_vars), priors_dict)
        p_priors = meandist2componentarray(priors_dict_p)
        u0p = ComponentVector(
            # merging to (priors merging to defaults)
            state = ComponentVector(ComponentVector(u0_default; u0_priors...); u0...),
            par = ComponentVector(ComponentVector(p_default; p_priors...); p...))
        problem = ODEProblem(system, system_num_dict(u0p.state, sys_num_dict), tspan,
            system_num_dict(u0p.par, sys_num_dict))
    end
end

"""
Put priors for `fixed, ranadd, ranmul, ranadd_σ, and ranmul_σ` into a ComponentVector.
The `indiv` priors can be indiv_id-specific They are setup separately in 
`setup_inversion`.     
"""
function setup_priors_pop(keys_fixed, keys_ranadd, keys_ranmul;
        inv_case::AbstractCrossInversionCase, scenario,
        priors_dict = get_case_priors_dict(inv_case, missing; scenario),
        priors_random_dict = get_case_priors_random_dict(inv_case; scenario))
    (;
        fixed = dict_to_cv(keys_fixed, priors_dict),
        ranadd = dict_to_cv(keys_ranadd, priors_dict),
        ranmul = dict_to_cv(keys_ranmul, priors_dict),
        ranadd_σ = dict_to_cv(keys_ranadd, priors_random_dict),
        ranmul_σ = dict_to_cv(keys_ranmul, priors_random_dict)
    )
end

"""
Take a ComponentArray of possibly multivariate distributions
and return a new ComponentArray of means of each distribution.
"""
meandist2componentarray = function (priors)
    # need to first create several ComponentVectors and then reduce
    # otherwise map on mixing Scalars and Vectors yields eltype any
    isempty(priors) && return (ComponentVector{Float64}())
    @chain priors begin
        map(keys(_)) do k
            ComponentVector(NamedTuple{(k,)}(Ref(mean(_[k]))))
        end
        reduce(vcat, _)
    end
end

"""
    get_indiv_parameters_from_priors(inv_case::AbstractCrossInversionCase,
            indiv_ids, mixed_keys;
            scenario, system,
            rng = StableRNG(234),
            priors_dict = get_case_priors_dict(inv_case, missing; scenario),
            priors_random_dict = get_case_priors_random_dict(inv_case; scenario),
            u0_default = ComponentVector(),
            p_default = ComponentVector(),
            )

Construct a DataFrame with parameters across sites with 
- fixed parameters corresponding to the mean of its prior
- ranadd parameters corresponding to mean modified, i.e. added to sampled 
  ranadd effects and its meta parameters
- ranmul parameters corresponding to mean modified, i.e. multiplied, by sampled 
  ranmul effects and its meta parameters
- individual parameters 

## Value
DataFrame with columns `indiv_id`, `u0` and `p`, with all states and parameters of the
given system as `ComponentVector` labelled by `get_state_labeled` and `get_par_labeled`.
"""
function get_indiv_parameters_from_priors(inv_case::AbstractCrossInversionCase;
        scenario = NTuple{0, Symbol}(),
        indiv_ids = get_case_indiv_ids(inv_case; scenario),
        mixed_keys = get_case_mixed_keys(inv_case; scenario),
        system_u0_p_default = get_case_inverted_system(inv_case; scenario),
        psets = setup_psets_mixed(mixed_keys; system = system_u0_p_default.system),
        rng = StableRNG(234),
        #priors_dict_indiv = get_priors_dict_indiv(inv_case, indiv_ids; scenario),
        priors_random_dict = get_case_priors_random_dict(inv_case; scenario)
)
    n_indiv = length(indiv_ids)
    tspans = fill((0.0, 0.0), n_indiv)
    df = DataFrame(
        indiv_id = collect(indiv_ids),
        problem = setup_indiv_problems(; inv_case, scenario, tspans, system_u0_p_default)
    )
    # make sure to use the same distribution across all individuals
    ranadd_dist_cv = get_ranadd_dist(
        sample_random_σ(rng, mixed_keys.ranadd, priors_random_dict))
    ranmul_dist_cv = get_ranmul_dist(
        sample_random_σ(rng, mixed_keys.ranmul, priors_random_dict))
    _resample_random = (problem) -> begin
        if isempty(mixed_keys.ranadd)
            ranadd_offset = ComponentVector()
        else
            ranadd = get_paropt_labeled(psets.ranadd, problem)
            ranadd_offset = map(dist -> Base.rand(rng, dist), ranadd_dist_cv)
            a = ranadd .+ ranadd_offset
            probo = remake(problem, a, psets.ranadd)
        end
        if isempty(mixed_keys.ranmul)
            ranmul_factor = ComponentVector()
        else
            ranmul = get_paropt_labeled(psets.ranmul, problem)
            ranmul_factor = map(dist -> Base.rand(rng, dist), ranmul_dist_cv)
            r = ranmul .* ranmul_factor
            probo = remake(problem, r, psets.ranmul)
        end
        (get_state_labeled(psets.popt, probo),
            get_par_labeled(psets.popt, probo),
            get_paropt_labeled(psets.popt, probo),
            ranadd_offset,
            ranmul_factor)
    end
    #tmp = _resample_random(df.problem[1]); length(tmp)
    DataFrames.transform!(df,
        [:problem] => DataFrames.ByRow(_resample_random) => [
            :u0, :p, :paropt, :indiv_ranadd, :indiv_ranmul])
    prob1 = df.problem[1]
    # for the first row remove the random effects modifications and stick to the mean
    # of the original problem
    df[1, [:u0, :p, :paropt]] .= (get_state_labeled(psets.popt, prob1),
        get_par_labeled(psets.popt, prob1), get_paropt_labeled(psets.popt, prob1))
    df[1, :indiv_ranadd] .= 0.0
    df[1, :indiv_ranmul] .= 1.0
    (; p_indiv = df[:, Not(:problem)], ranadd_dist_cv, ranmul_dist_cv)
end

# function check_equal_across_indiv(kc, mixed_indiv)
#     tmp = ComponentMatrix(hcat((c[kc] for c in mixed_indiv)...),
#         first(getaxes(mixed_indiv[1][kc])), FlatAxis())
#     rows_cols_equal = map(x -> all(x .== first(x)), eachrow(tmp))
#     all(rows_cols_equal) && return ()
#     @warn("Expected $kc effects priors to be equal across individuals, "*
#     "but means differed in rows $(findall(rows_cols_equal)): $(tmp)")
# end

function get_priors_dict_indiv(inv_case, indiv_ids; scenario)
    Dict(id => get_case_priors_dict(inv_case, id; scenario) for
    id in indiv_ids)
end
