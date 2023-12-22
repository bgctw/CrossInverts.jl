# move to MTKHelpers?

"""
    setup_mtk_tools(system, u0, p, popt_names, subsystems; 
        tspan=(0,100), problemupdater=NullProblemUpdater)

Setup a collection of common tools to work with a ModelingToolkit ODESystmes.

# Arguments
- system: ODESystem to work with
- u0: ComponentVecotr of initial states
- p: ComponentVector of parameters
- popt_names: keys in u0 or p to be updated from a parameter vector
- subsystems: tuple of ODESystems that hold the Nums (see embed_system)

# Value
NamedTuple with entries
- pset: ProblemParameterSetter for setting popt_names into problem
- u_map: mapping of quantities given with CompoenentArray u0 to problem.u0
- p_map: mapping of quantities given with CompoenentArray u0 to problem.p
- problemupdter: restate the optional argument
- problem: ODEProblem constructed from the system
- system: restate the arguemt

When constructing an ODEProblem from an ODESystem, the order of entries
in problem.u0 and problem.p are undefined. 
Hence, one needs pset, u0_map, and p_map to update specific parts of the problem.
E.g. update the initial state of the problem by `remake(prob, u0=myu0[u_map])`.
"""
function setup_mtk_tools(system, u0, p, popt_names;
        tspan = (0, 100), problemupdater = NullProblemUpdater())
    _dict_nums = get_system_symbol_dict(system)
    problem = ODEProblem(system,
        system_num_dict(u0, _dict_nums), tspan, system_num_dict(p, _dict_nums))
    pset = ODEProblemParSetter(system, strip_deriv_num.(popt_names))
    u_map = StaticArrays.SVector{length(u0)}(get_u_map(keys(u0), pset))
    p_map = StaticArrays.SVector{length(p)}(get_p_map(keys(p), pset))
    (; pset, u_map, p_map, problemupdater, problem)
end

"""
    setup_psets_fixed_random_indiv(system, keys_popt, keys_opt_fixed, keys_opt_random)

Setup the ProblemParSetters for given system and parameter names.

Only the entries in keys_opt_fixed and keys_opt_random that are actually occuring
in popt_names are set.
The indiv parameters are the difference set between popt_names and the others.

Returns a NamedTuple with `ODEProblemParSetter` for `fixed`, `random`, and `indiv` parameters.
"""
function setup_psets_fixed_random_indiv(system, keys_popt, keys_opt_fixed, keys_opt_random)
    keys_opt_fixed1 = intersect(keys_popt, keys_opt_fixed)
    keys_opt_random1 = intersect(keys_popt, keys_opt_random)
    keys_opt_site1 = setdiff(keys_popt, union(keys_opt_fixed1, keys_opt_random1))
    psets = (;
        fixed = ODEProblemParSetter(system, CA.Axis(keys_opt_fixed1)),
        random = ODEProblemParSetter(system, CA.Axis(keys_opt_random1)),
        indiv = ODEProblemParSetter(system, CA.Axis(keys_opt_site1)))
    return psets
end

function setup_psets_fixed_random_indiv(system, chn::MCMCChains.Chains)
    @error "not fully implemented yet."
    keys_opt = extract_popt_names(chn)
    setup_psets_fixed_random_indiv(system,
        keys_opt[(:par_names, :keys_opt_fixed, :keys_opt_random)]...)
end
