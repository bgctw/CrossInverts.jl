#TODO define priors
# models the multiplicative standard distribution σstar of the random effects
# 95% of the probability mass will be in (μ/σstar^2, μ*σstar^2)
# hence for σstar=2 (the mean of the default case: 1+1) in (1/4, 4)*μ
# should be >=1 -> using a shifted Exponential for 
# for the start use expected σstar=3: 1-Shifted-Expoenential(2)
function get_priors_σ_dict(default = Distributions.AffineDistribution(1, 1, Exponential(2));
        kwargs...)
    priors_default = Dict([:s₊a_E, :s₊k_mN_L, :s₊k_L, :s₊k_R, :s₊k_LP] .=> default)
    priors_kwargs = Dict(kwargs...)
    merge(priors_default, priors_kwargs)
end

# loads initial parameters and u0 from site_fits.jld2, saved from optim.jl
#@run 
#chn_popt_u, probci, psetci = fsample(indiv_id, targetlim, scenario; n_sample=10, n_burnin=0); chn = chn_popt_u;
function sample_slc_cross(df_site_u0_p, targetlim, scenario;
        keys_opt_fixed = (), keys_opt_random = (), priors_σ_dict = get_priors_σ_dict(),
        n_sample = 750, n_burnin = 100, kwargs...)
    df = copy(df_site_u0_p)
    add_sitedata_obs_tools!(df, targetlim, scenario) # reflects u0 and p in tools.problem
    tools1 = first(df.tools)
    popt_names = symbols_paropt(tools1.psetci)
    psets = setup_psets_fixed_random_indiv(keys_opt_fixed,
        keys_opt_random; system = tools1.system, popt = missing) #TODO #popt_names )
    popt0 = get_popt_from_tools(df.tools, psets)
    priors_opt = get_priors_opt_from_tools(df.tools, psets, priors_σ_dict)
    # function barrier so that all types are know inside
    #  - do not pass DataFrame, but passsingle columns to specialize
    #  - pass df.tools as tuple, so that length can be inferred
    sim_sols_probs = gen_sim_sols_probs(; tools = (df.tools...,), solver, psets)
    tmodel = gen_model_cross(;
        #indiv_id=(df.indiv_id...,), 
        tools = (df.tools...,), obs = df.obs, priors_opt,
        scenario, solver, psets, sim_sols_probs)
    # vi = Turing.VarInfo(tmodel); spl = Turing.SampleFromPrior(); tmp = tmodel(vi, spl)
    #return((;popt0, psets.fixed, psets.random, psets.indiv, sim_sites, parkeys_dict))
    nthreads = Threads.nthreads()
    #n_burnin = 0; n_sample = 10
    chn = chn_orig = ((n_sample + n_burnin < 100) || (nthreads == 1)) ?
                     Turing.sample(tmodel, Turing.NUTS(n_burnin, 0.65, init_ϵ = 1e-2),
        n_sample, init_params = collect(popt0)) :
                     Turing.sample(tmodel, Turing.NUTS(n_burnin, 0.65, init_ϵ = 1e-2),
        Turing.MCMCThreads(),
        n_sample, nthreads, init_params = repeat([collect(popt0)], nthreads))
    # rename columns to symbols of optimized parameters and sort into sections
    parkeys_dict = get_parameter_keys(psets; n_indiv = nrow(df))
    #Maybe: think of actual dictionary for the case the order in chn is not stable
    paropt_sections = (:fixed, :random, :random_σstar, :indiv, :indiv_random)
    parkeys_dict_p_flat = OrderedDict(x => vec(parkeys_dict[x]) for x in paropt_sections)
    parkeys = vcat(values(parkeys_dict_p_flat)...)
    chn = replacenames(chn_orig, Dict(names(chn, :parameters) .=> parkeys))
    chn = set_section(chn,
        merge(parkeys_dict_p_flat, Dict(:internals => names(chn, :internals))))
    # add uends to chain by callimg sim_sites
    sim_sites = gen_sim_sols(sim_sols_probs)
    u0_chains = compute_u0_chains(chn_orig, sim_sites; popt0, keys_u0 = parkeys_dict[:u0],
        tend = tools1.problem.tspan[end])
    chn = hcat(MCMCChains.resetrange(chn), u0_chains...)
    # construct ComponentArray with Axis for both, popt and u
    u0l = CA.ComponentArray(reshape(Array(Chains(chn, :u0)[1, :, 1]),
            size(parkeys_dict[:u0])),
        (axis_state(psets.fixed), CA.FlatAxis()))
    poptul = CA.ComponentArray(popt = popt0, u = u0l)
    # parkeys_dict holds matrices, handled easier than chn.sections
    (; chn, poptul, tools = df.tools, psets, parkeys_dict, paropt_sections)
end

ftmp = () -> begin
    rename(chn)
    Symbol.("σ_" .* string.(symbols_paropt(psets.random)))
end

i_call_once = () -> begin
    vi = Turing.VarInfo(tmodel)
    spl = Turing.SampleFromPrior()
    tmodel(vi, spl)
    #using Cthulhu
    #@descend_code_warntype tmodel(vi, spl)
    # from infiltrate in sppfit
    #@descend_code_warntype setprob(probci, u0v, popt, psetci, parupdater) 
    #@exfiltrate probci u0v popt scenario system psetci problemupdater
    #@code_warntype setprob(safehouse.probci, safehouse.u0v, safehouse.popt, safehouse.psetci, safehouse.parupdater) 
end

"""
Add the following columns DataFrame of with a row for each indiv_id.
- sitedata: DataFrameRow of litter and stocks for given indiv_id for :top30 (see get_sitedata_df)
- obs: NamedTuple of cstocks, cn, cp
- tools: return of setup_tools_indiv for indiv_id, targetlim, scenario, u0, p 

The following columns need to be present already: indiv_id, u0, p.
The initial parameters u0 and parameters p need to be given as ComponentVectors.
"""
function add_sitedata_obs_tools!(df_site_u0_p, targetlim, scenario)
    df = df_site_u0_p #shorthand
    if :sitedata ∉ names(df)
        sd_df = subset(get_sitedata_df(), :layer => ByRow(==(:top30)))
        fsitedata = (indiv_id) -> sd_df[findfirst(==(indiv_id), sd_df.indiv_id), :]
        transform!(df, :indiv_id => ByRow(fsitedata) => :sitedata)
    end
    f_obs = (sd) -> SLVector(cstocks = sd[:SOC], cn = sd[:CN_SOM], cp = sd[:CP_SOM])
    transform!(df, :sitedata => ByRow(f_obs) => :obs)
    _, _, popt_names = setup_popt_scenario(first(df.indiv_id),
        targetlim,
        scenario;
        sitedata = first(df.sitedata))
    f_tools = (indiv_id, u0, p) -> setup_tools_indiv(indiv_id,
        targetlim,
        scenario,
        u0,
        p,
        popt_names)
    transform!(df, [:indiv_id, :u0, :p] => ByRow(f_tools) => :tools)
end

function get_popt_from_tools(tools, psets)
    popt_indiv = hcat((get_paropt_labeled(psets.indiv, tools_site.problem) for
                       tools_site in tools)...) |>
                 x -> CA.ComponentArray(x, (axis_paropt(psets.indiv), CA.FlatAxis()))
    # for the fixed parameters take the mean across the popt-sites
    popt_fixed_sites = hcat((get_paropt_labeled(psets.fixed, tools_site.problem) for
                             tools_site in tools)...) |>
                       x -> CA.ComponentArray(x, (axis_paropt(psets.fixed), CA.FlatAxis()))
    popt_fixed = CA.ComponentVector(vec(mean(convert(Array, popt_fixed_sites); dims = 2)),
        axis_paropt(psets.fixed))
    # for the random effecs also take the mean across popt-sites
    popt_random_sites = hcat((get_paropt_labeled(psets.random, tools_site.problem) for
                              tools_site in tools)...) |>
                        x -> CA.ComponentArray(x,
        (axis_paropt(psets.random), CA.FlatAxis()))
    popt_random = CA.ComponentVector(vec(mean(convert(Array, popt_random_sites), dims = 2)),
        axis_paropt(psets.random))
    # the random offsets are multiplicator hence compute r_eff = indiv_id/r_mean for each col
    popt_indiv_random = hcat((r_site ./ popt_random for
                              r_site in eachcol(popt_random_sites))...) |>
                        x -> CA.ComponentArray(x, CA.getaxes(popt_random_sites))
    popt_random_σstar = similar(popt_random)
    popt_random_σstar .= 3.0 # cf95 between 1/9 and 9x
    popt_siteoffsets = popt_indiv_random ./ popt_random
    #emptyCA = CA.ComponentVector{eltype(popt_fixed)}()
    # unfortunately the label-information for indiv and indiv_random is lost here
    popt0 = CA.ComponentArray(fixed = popt_fixed, random = popt_random,
        random_σstar = popt_random_σstar,
        indiv = popt_indiv, indiv_random = popt_indiv_random)
end

"""
Return priors whose order matches the fixed, random, and indiv_id parameters.
Gets the priors for fixed and ranodm parameters from first tools entry,
While the indiv_id priors are mapped from each entry in the tools-vector.

`priors_σ_dict` can be obtained from function get_priors_σ_dict.
"""
function get_priors_opt_from_tools(tools, psets, priors_σ_dict)
    tools1 = first(tools)
    priors_opt = (;
        BtoSOC = tools1.priors_dict[:s₊BtoSOC],
        fixed = [tools1.priors_dict[p] for p in symbols_paropt(psets.fixed)],
        random = [tools1.priors_dict[p] for p in symbols_paropt(psets.random)],
        random_σstar = [priors_σ_dict[p] for p in symbols_paropt(psets.random)],
        indiv = map(tools) do tools
            # maybe different for each indiv_id
            [tools.priors_dict[p] for p in symbols_paropt(psets.indiv)]
        end)
end

function gen_model_cross(;
        tools, obs, priors_opt, sim_sols_probs, scenario, psets = psets, solver = solver)
    fLogger = EarlyFilteredLogger(current_logger()) do log
        #@show log
        !(log.level == Logging.Warn && log.group == :integrator_interface)
    end
    n_indiv = length(tools)
    #
    gen_tmodel_cross = let pdf_deficit = Exponential(0.001),
        #prior_dist = prior_dist, 
        scenario = scenario,
        n_indiv = n_indiv, tools = tools,
        psets = psets,
        solver = solver,
        priors_opt = priors_opt,
        sim_sols_probs = sim_sols_probs

        Turing.@model function tmodel_cross(obs, ::Type{T} = Float64) where {T}
            npfix = count_paropt(psets.fixed)
            nprand = count_paropt(psets.random)
            npsite = count_paropt(psets.indiv)
            #i0_fixed = 0
            # i0_μb = npfix
            # i0_σb = i0_μb + nprand
            # i0_sites = i0_σb + nprand
            # i0_siteoffsets = i0_sites + n_indiv*npsite
            #npopt = i0_siteoffsets + n_indiv*nprand
            #local popt = Vector{T}(undef, npopt)
            # T == Float64 || @infiltrate
            # Turing.: replace all popt[]= by popt[]~r
            #fixed = Vector{T}(undef, npfix)
            fixed = StaticArrays.MVector{npfix, T}(undef)
            for (i, r) in enumerate(priors_opt.fixed)
                fixed[i] ~ r
                #popt[i0_fixed+i] ~ r # = popt0[i0_fixed+i]
            end
            #random = Vector{T}(undef, nprand)
            random = StaticArrays.MVector{nprand, T}(undef)
            for (i, r) in enumerate(priors_opt.random)
                random[i] ~ r # = popt0[i0_μb+i] #mean(r)
                #popt[i0_μb+i] ~ r # = popt0[i0_μb+i] #mean(r)
            end
            #prand_σstar = Vector{T}(undef, nprand)
            prand_σstar = StaticArrays.MVector{nprand, T}(undef)
            for (i, r) in enumerate(priors_opt.random_σstar)
                #popt[i0_σb+i] ~r # = popt0[i0_σb+i] #mean(r)
                prand_σstar[i] ~ r # = popt0[i0_σb+i] #mean(r)
            end
            # indiv = Matrix{T}(undef, npsite, n_indiv)
            # indiv_random = Matrix{T}(undef, nprand, n_indiv)
            indiv = StaticArrays.MMatrix{npsite, n_indiv, T}(undef)
            indiv_random = StaticArrays.MMatrix{nprand, n_indiv, T}(undef)
            for i_indiv in 1:n_indiv
                # i0_isite = i0_sites + (i_indiv-1)*count_paropt(psets.indiv)
                for (i, r) in enumerate(priors_opt.indiv[i_indiv])
                    indiv[i, i_indiv] ~ r # = popt0[i0_isite+i] #mean(r)
                    # popt[i0_isite+i] ~r # = popt0[i0_isite+i] #mean(r)
                end
                #i0_isite_offset = i0_siteoffsets + (i_indiv-1)*count_paropt(psets.random)
                for i in 1:count_paropt(psets.random)
                    #σ = log(prand_σstar[i])  # Lognormal-σ from multiplicative stddev σ*=e^σ
                    indiv_random[i, i_indiv] ~ fit(LogNormal(1, Σstar(exp(prand_σstar[i])))) # = popt0[i0_isite_offset+i] 
                end
            end
            #poptl = CA.ComponentVector(popt, first(getaxes(popt0)))
            #poptl = CA.ComponentVector(
            #   vcat(fixed, random, prand_σstar, vec(indiv), vec(indiv_random)), first(getaxes(popt0)))
            #@show poptl
            res_sim = sim_sols_probs(fixed,
                random,
                indiv,
                indiv_random;
                saveat = first(tools).problem.tspan[end])
            for i_indiv in 1:n_indiv
                # not in 1.6 (; sol, problem_opt) = res_sim[i_indiv]
                (sol, problem_opt) = res_sim[i_indiv]
                #!is_dual && @show popt, sol.retcode
                if !SciMLBase.successful_retcode(sol.retcode)
                    Turing.@addlogprob! -Inf
                    return
                end
                local parl = label_par(first(tools).psetci, problem_opt.p) #@inferred label_par(psetci, problem_opt.p)
                # for accessing solution at 100 time points need to store full solution
                #   here (saveat) only the final solution was stored to save allocations
                pred = getobs(sol, parl) #getobs_avg(sol, parl) #@inferred getobs(sol, parl)
                # Turing.
                # obst = (;zip((:cstocks,:cn,:cp),obs[:,i_indiv])...)
                # dist_pred = get_dist_obs(pred, obst) #@inferred get_dist_obs(pred, obs)
                # obs[1,i_indiv] ~ dist_pred.cstocks
                # obs[2,i_indiv] ~ dist_pred.cn
                # obs[3,i_indiv] ~ dist_pred.cp
                dist_pred = get_dist_obs(pred, obs[i_indiv])
                obs[i_indiv].cstocks ~ dist_pred.cstocks
                obs[i_indiv].cn ~ dist_pred.cn
                obs[i_indiv].cp ~ dist_pred.cp
                # logpdf(dist_pred.cstocks, obs[i_indiv].cstocks)
                # logpdf(dist_pred.cn, obs[i_indiv].cn)
                # logpdf(dist_pred.cp, obs[i_indiv].cp)
                # prepare extracting u0 with generated_quantities
                # sharply decreasing probability of plant inorganic P and N uptake + imbalanc is lower than  litter production (may happen at very low I_P)
                deficit_u_PlantP = max(0.0,
                    parl.pl₊i_L0 / parl.pl₊β_Pi0 -
                    (sol[s.u_PlantP][end] + parl.pl₊imbalance_P0))
                deficit_u_PlantN = max(0.0,
                    parl.pl₊i_L0 / parl.pl₊β_Ni0 -
                    (sol[s.u_PlantN][end] + parl.pl₊imbalance_N0))
                prob_u_PlantP = logpdf(pdf_deficit, deficit_u_PlantP)
                prob_u_PlantN = logpdf(pdf_deficit, deficit_u_PlantN)
                # penalize negative root uptake
                neg_root_uptake = max(0.0, -sol[s.u_PlantP][end])
                prob_positive_root_uptake = logpdf(pdf_deficit, neg_root_uptake)
                Turing.@addlogprob! prob_u_PlantP + prob_u_PlantN +
                                    prob_positive_root_uptake
                if :prior_B ∈ scenario
                    # also constrain possible microbial biomass (wide prior but > 0)
                    # to exclude vanishing microbial biomass levels
                    mBtoSOC = sol[s.B][end] / (sol[s.L][end] + sol[s.R][end])
                    logpdf_mBtoSOC = logpdf(priors_opt.BtoSOC, mBtoSOC)
                    Turing.@addlogprob! logpdf_mBtoSOC
                end
            end # for i_indiv
            :return_from_tmodel_cross
        end # function      
    end # let
    #
    gen_tmodel_cross(obs)
end

"""
    get_parameter_keys(psets; n_indiv)

Get all the parameters given the ProblemParameterSetters (fixed, random, indiv_id)
given in NamedTuple psets for Dict entries
- :fixed    
- :random    
- :random_σstar
- :indiv
- :indiv_random
- :u0
"""
function get_parameter_keys(psets; n_indiv)
    parkeys_dict = OrderedDict(:fixed => collect(symbols_paropt(psets.fixed)),
        :random => collect(symbols_paropt(psets.random)),
        :random_σstar => [Symbol("σstar_" * string(pname))
                          for pname in symbols_paropt(psets.random)],
        :indiv => [Symbol("s" * string(x[2]) * "_" * string(x[1]))
                   for x in IterTools.product_distribution(symbols_paropt(psets.indiv),
            1:n_indiv)],
        :indiv_random => [Symbol("r" * string(x[2]) * "_" * string(x[1]))
                          for x in IterTools.product_distribution(symbols_paropt(psets.random),
            1:n_indiv)],
        :u0 => [Symbol("u0" * string(x[2]) * "_" * string(x[1]))
                for x in IterTools.product_distribution(symbols_state(psets.fixed),
            1:n_indiv)])
    return parkeys_dict
end

function compute_u0_chains(chn_orig, sim_sites; popt0, keys_u0, tend)
    u0s = mapslices(reshape(Array(chn_orig), size(Chains(chn_orig, :parameters)));
        dims = 2) do x
        # TODO Main.@infiltrate_main            
        poptl = CA.ComponentArray(x, CA.getaxes(popt0))
        #sols = sim_sites(poptl; tspan=(0,0)); sols[1].t
        sols = sim_sites(poptl; saveat = tend)
        #uend = map(sol -> label_state(psets.fixed, sol[:,end]), sols)
        uends = map(y -> y[end], sols)
    end
    # to be save for keys_u0 given in flat vector, reshape to matrix
    n_indiv = size(u0s, 2)
    n_u0 = length(keys_u0)
    keys_u0_m = reshape(keys_u0, (Integer(n_u0 / n_indiv), n_indiv))
    u0_chains = map(axes(u0s, 2)) do i_indiv
        u0i = mapslices(x -> x[i_indiv], u0s; dims = 2)
        u0names = keys_u0_m[:, i_indiv]
        Chains(u0i, u0names, Dict(:u0 => u0names))
    end
end

function extract_popt_names(chn::MCMCChains.Chains)
    keys_opt_fixed = names(chn, :fixed)
    keys_opt_random = names(chn, :random)
    keys_opt_sites = [replace(x, r"^s1_" => "") |> Symbol
                      for x in string.(names(chn, :indiv)) if occursin(r"^s1_", x)]
    par_names = union(keys_opt_fixed, keys_opt_random, keys_opt_sites)
    (; par_names, keys_opt_fixed, keys_opt_random, keys_opt_sites)
end

function extract_u0_names(chn::MCMCChains.Chains)
    [replace(x, r"^u01_" => "") |> Symbol
     for x in string.(names(chn, :u0)) if occursin(r"^u01_", x)]
end

"""
Extract chain objects, where chains are first stacked and
then unstacked so that different chains represent different indiv.
Names of corresponding objects omit prefixes and correspond to Symbols
in the system. To avoid duplicated names, e.g. for factor and coefficient,
several Chains objects are returned.

Returns a NamedTuple of 
- `chn_sites_popt`: with sections `indiv_id` (parameters free between sites) and 
  `site_coef` (parameters with random effects: mean multiplied by indiv_id specific factor)
-  `chn_sites_random`: indiv_id specific random factors
-  `chn_sites_u0`: initial states
"""
function extract_chains_site(chn::MCMCChains.Chains)
    keys_u0 = extract_u0_names(chn)
    keys_opt = extract_popt_names(chn)
    n_u0 = length(keys_u0)
    n_u0site = length(names(chn, :u0))
    n_indiv = n_u0site ÷ n_u0
    n_indiv * n_u0 == n_u0site ||
        error("Expected u0-section to hold a multiple of u0: $(n_u0), but got $(n_u0site).")
    #
    a_fixed = Array(chn, :fixed)
    a_random = Array(chn, :random)
    nsample = size(a_fixed, 1)
    tmp = Array(chn, :indiv)
    a_sites = reshape(tmp, (nsample, size(tmp, 2) ÷ n_indiv, n_indiv))
    tmp = Array(chn, :indiv_random)
    a_sites_random = reshape(tmp, (nsample, size(tmp, 2) ÷ n_indiv, n_indiv))
    tmp = Array(chn, :u0)
    a_site_coef = a_random .* a_sites_random
    a_u0 = reshape(tmp, (nsample, size(tmp, 2) ÷ n_indiv, n_indiv))
    #
    popt = hcat(MCMCChains.Chains(a_sites, collect(keys_opt.keys_opt_sites),
            (:indiv_id => keys_opt.keys_opt_sites,)),
        MCMCChains.Chains(a_site_coef, collect(keys_opt.keys_opt_random),
            (:site_coef => keys_opt.keys_opt_sites,)))
    random = MCMCChains.Chains(a_sites_random, collect(keys_opt.keys_opt_random))
    u0 = MCMCChains.Chains(a_u0, collect(keys_u0))
    (; popt, random, u0)
end

function add_obs_cross(lc, chn::MCMCChains.Chains, sites, var_obs;
        tools = NamedTuple((indiv_id => setup_tools_indiv(indiv_id, lc...)
                            for indiv_id in sites)),
        kwargs...)
    add_obs_cross(chn, var_obs; tools, kwargs...)
end

function add_obs_cross(chn::MCMCChains.Chains, var_obs;
        tools,
        chn_sites = extract_chains_site(chn))
    var_obs_pairs = map(var_obs) do item
        item isa Num && return (symbol(item) => item)
        item isa Pair && return (item)
        error("expected each item of var_obs to be a Pair(:Symbol=>Num) or a Num, but got $(item) of type $(typeof(item))")
    end
    var_obs_vec = collect(last.(var_obs_pairs))
    var_obs_keys = collect(first.(var_obs_pairs))
    #
    sites = keys(tools)
    #TODO psets = setup_psets_fixed_random_indiv(first(tools).system, chn)
    pset_u0 = ODEProblemParSetter(first(tools).system, extract_u0_names(chn))
    sim_sols_probs = gen_sim_sols_probs(; tools, solver, psets)
    #
    b_fixed = Array(chn, :fixed)
    b_random = Array(chn, :random)
    b_sites = Array(chn_sites.popt, :indiv_id; append_chains = false) |>
              RecursiveArrayTools.VectorOfArray
    b_sites_random = Array(chn_sites.random; append_chains = false) |>
                     RecursiveArrayTools.VectorOfArray
    b_u0 = Array(chn_sites.u0; append_chains = false) |>
           RecursiveArrayTools.VectorOfArray
    #
    #i_sample = 1
    nsample = size(b_fixed, 1)
    n_indiv = length(sites)
    ra = map(1:nsample) do i_sample
        sols = map(x -> x.sol,
            sim_sols_probs(b_fixed[i_sample, :], b_random[i_sample, :],
                b_sites[i_sample, :, :], b_sites_random[i_sample, :, :];
                tspan = (0.0, 0.0),
                pset_u0, u0 = b_u0[i_sample, :, :]))
        #label_state(first(tools).psetci, sols[1][end])
        ra_sample = map(i_site -> getindex.(Ref(sols[i_site]), var_obs_vec, 1),
            1:n_indiv) |>
                    RecursiveArrayTools.VectorOfArray
    end |> RecursiveArrayTools.VectorOfArray
    #
    chns_obs_site = map(1:n_indiv) do i_site
        chn_obs_site = MCMCChains.Chains(collect(ra[:, i_site, :]'),
            var_obs_keys
            #Dict(:obs => var_obs_keys)
        )
    end
    chn_obs_sites = cat(chns_obs_site..., dims = 3)
end
