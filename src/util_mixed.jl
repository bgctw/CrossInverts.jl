"""
    setup_inversion(inv_case::AbstractCrossInversionCase; scenario = NTuple{0, Symbol}())

Calls all the functions for specific [`AbstractCrossInversionCase`](@ref) to setup the
inversion.

Returns a `NamedTuple` with entries: `(; system, indiv_info, pop_info)`
See [`setup_tools_mixed`](@ref) for a description of `indiv_info` and `pop_info`.
Additional components of `pop_info` are
- `mixed_keys`: The optimized parameters and their distribution across individual 
  as returned by [`get_mixed_keys]`(@ref)
- `indiv_ids`: A tuple of ids (Symbols) of the individuals taking part in the inversion, 
  as returned by [`get_indiv_ids]`(@ref)

"""
function setup_inversion(inv_case::AbstractCrossInversionCase; scenario = NTuple{0, Symbol}())
    (;system, u0_default, p_default) = get_inverted_system(inv_case; scenario)
    mixed_keys = get_mixed_keys(inv_case; scenario)
    indiv_ids = get_indiv_ids(inv_case; scenario)
    p_indiv = get_indiv_parameters_from_priors(inv_case; 
        scenario, indiv_ids, mixed_keys, system, u0_default, p_default)
    (; pop_info, indiv_info) = setup_tools_mixed(p_indiv; 
        inv_case, scenario, system, mixed_keys)
    pop_info = (;mixed_keys, indiv_ids, pop_info..., )    
    (; system, indiv_info, pop_info)
end

"""
setup_tools_mixed(p_indiv::DataFrame;
        inv_case, scenario = NTuple{0, Symbol}(), 
        system, mixed_keys,
        psets = setup_psets_mixed(inv_case; scenario, mixed_keys, system))

Given `inv_case`, the keys for different mixed effects, individual state 
(`u0`, `p` given with p_indiv), 
and individual priors, sets up NamedTuple of 
- `mixed`: mixed effects `NamedTuple(fixed, random, indiv, indiv_random)`
  from individual's states and parameters
- `df`: DataFrame `p_indiv` extended by columns
  - `paropt`: optimized parameters extracted from indiviudals state and parameters
  - `tools`: tools initialized for each site (see `setup_tools_indiv`)
- `psets`: `NTuple{ODEProblemParSetter}` for each mixed component
- `priors_pop`: `ComponentVector` of priors on population level (fixed, random, random_σ)
- `sample0`: ComponentVector of an initial sample
  This can be used to name (attach_axis) a sample from MCMCChains object 
"""
function setup_tools_mixed(p_indiv::DataFrame;
        inv_case, scenario = NTuple{0, Symbol}(), 
        system, mixed_keys,
        psets = setup_psets_mixed(inv_case; scenario, mixed_keys, system))
    df = copy(p_indiv)
    u0_numdict = system_num_dict(df.u0[1], system)
    p_numdict = system_num_dict(df.p[1], system)
    prob = ODEProblem(system, u0_numdict, (0,2), p_numdict);
    _extract_paropt = let prob=prob, system=system, pset = psets.popt
        (u0, p) -> begin
            u0_numdict = system_num_dict(u0, system)
            p_numdict = system_num_dict(p, system)
            probo = remake(prob, u0 = u0_numdict, p = p_numdict)
            flatten1(get_paropt_labeled(pset, probo))
        end        
    end
    DataFrames.transform!(df,
        [:u0, :p] => DataFrames.ByRow(_extract_paropt) => :paropt)
    mixed = extract_mixed_effects(psets, df.paropt)
    priors_pop = setup_priors_pop(keys(mixed.fixed), keys(mixed.random); inv_case, scenario)
    _setuptools = (indiv_id, u0, p) -> setup_tools_indiv(indiv_id; inv_case, scenario,
        system, u0, p, keys_indiv = mixed_keys.indiv)
    #_tools = _setuptools(df.indiv_id[1], df.u0[1], df.p[1])
    DataFrames.transform!(df,
        [:indiv_id, :u0, :p] => DataFrames.ByRow(_setuptools) => :tools)
    sample0 = get_init_mixedmodel(psets,
        df.paropt,
        priors_pop.random_σ;
        indiv_ids = df.indiv_id)
    effect_pos = MTKHelpers.attach_axis(1:length(sample0), MTKHelpers._get_axis(sample0))
    problemupdater = get_problemupdater(inv_case; system, scenario)
    #
    pop_info = (;mixed, psets, problemupdater, priors_pop, sample0, effect_pos)
    (; pop_info, indiv_info = df)
end

"""
    setup_psets_mixed(mixed_keys; system, popt)
    
    setup_psets_mixed(inv_case::AbstractCrossInversionCase;
        scenario, mixed_keys, system,
        priors_dict = get_priors_dict(inv_case, missing; scenario),)

Creates the `ODEProblemUpdaters` given several optimized parameters and a system.
Creates a separate Updater for each class of optimized parameters in mixed_keys.

The second variant creates the optimized parameters from the means of 
the prior distributions, obtained for given CrossInversionCase.

"""
function setup_psets_mixed(inv_case::AbstractCrossInversionCase;
        scenario, mixed_keys, system)
    priors_dict = get_priors_dict(inv_case, missing; scenario)
    mean_priors_mixed = mean_priors(; mixed_keys, priors_dict, system)
    setup_psets_mixed(mixed_keys; system, mean_priors_mixed.popt)
end
function setup_psets_mixed(mixed_keys; system, popt)
    gen = ((kc, ODEProblemParSetter(system, popt[mixed_keys[kc]])) for
           kc in keys(mixed_keys))
    psets = (; gen..., popt = ODEProblemParSetter(system, popt))
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
    gen = (begin
        priors_k = dict_to_cv(_keys, priors_dict)
        meandist2componentarray(priors_k)
    end
           for (_comp, _keys) in pairs(mixed_keys))
    ntup = (; zip(keys(mixed_keys), gen)...)
    popt = vcat_statesfirst(ntup...; system)
    (; ntup..., popt)
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
