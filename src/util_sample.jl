function gen_model_cross(;
        inv_case::AbstractCrossInversionCase, tools, priors_pop, sim_sols_probs,
        scenario, psets, solver)
    fLogger = EarlyFilteredLogger(current_logger()) do log
        #@show log
        !(log.level == Logging.Warn && log.group == :integrator_interface)
    end
    n_indiv = length(tools)
    sys_num_dict = get_system_symbol_dict(get_system(first(tools).problem))
    #obs = extract_stream_obsmatrices(;tools)
    obs = map(t -> t.sitedata, tools)
    obs_target = map(obs) do obs_site
        map(obs_site_stream -> obs_site_stream.obs, obs_site)
    end
    #
    gen_tmodel_cross = let pdf_deficit = Exponential(0.001),
        #prior_dist = prior_dist, 
        scenario = scenario,
        n_indiv = n_indiv, tools = tools,
        psets = map(get_concrete, psets),
        solver = solver,
        priors_pop = priors_pop,
        priors_indiv = [t.priors_indiv for t in tools],
        sim_sols_probs = sim_sols_probs,
        obs = obs,
        # assume all sites/indiv have same streams
        streams = keys(first(obs)),
        dtypes = (;
            zip(streams,
                (get_obs_uncertainty_dist_type(inv_case, s; scenario) for s in streams))...)
        #saveat = union(map_keys(stream -> stream.t, obs)),
        stream_nums = (;
            zip(streams,
                Symbolics.scalarize.(getindex.(Ref(sys_num_dict), streams)))...)

        Turing.@model function tmodel_cross(obs_target, ::Type{T} = Float64) where {T}
            npfix = count_paropt(psets.fixed)
            nprand = count_paropt(psets.random)
            npsite = count_paropt(psets.indiv)
            fixed_nl = StaticArrays.MVector{npfix, T}(undef)
            #fixed_nl = Vector{T}(undef, npfix)
            fixed = fixed1 = label_paropt_flat1(psets.fixed, fixed_nl)
            #k = first(keys(fixed))
            for k in keys(fixed)
                fixed[k] ~ priors_pop.fixed[k]
                #fixed[k] = rand(priors_pop.fixed[k])
            end
            #random = Vector{T}(undef, nprand)
            random_nl = StaticArrays.MVector{nprand, T}(undef)
            random = label_paropt_flat1(psets.random, random_nl)
            #k = first(keys(random))
            for k in keys(random)
                random[k] ~ priors_pop.random[k]
                #random[k] = rand(priors_pop.random[k])
            end
            #prand_σ = Vector{T}(undef, nprand)
            prand_σ_nl = StaticArrays.MVector{nprand, T}(undef)
            prand_σ = label_paropt_flat1(psets.random, prand_σ_nl)
            #k = first(keys(prand_σ))
            for k in keys(prand_σ)
                prand_σ[k] ~ priors_pop.random_σ[k]
                #prand_σ[k] = rand(priors_pop.random_σ[k])
            end
            # indiv = Matrix{T}(undef, npsite, n_indiv)
            # indiv_random = Matrix{T}(undef, nprand, n_indiv)
            #indiv1_l = label_paropt_flat1(psets.indiv, 1:count_par(psets.indiv))
            #template = label_paropt_flat1(psets.indiv, 1:count_paropt(psets.indiv))
            ax_indiv = axis_paropt_flat1(psets.indiv)
            indiv_nl = StaticArrays.MMatrix{npsite, n_indiv, T}(undef)
            indiv = ComponentArray(getdata(indiv_nl), ax_indiv, FlatAxis())
            indiv_random_nl = StaticArrays.MMatrix{nprand, n_indiv, T}(undef)
            indiv_random = ComponentArray(getdata(indiv_random_nl),
                first(getaxes(random)), FlatAxis())
            #i_indiv = 1
            for i_indiv in 1:n_indiv
                prior_indiv = priors_indiv[i_indiv]
                #k = first(keys(prior_indiv))
                for k in keys(prior_indiv)
                    indiv[k, i_indiv] ~ prior_indiv[k]
                    #indiv[k,i_indiv] = rand(prior_indiv[k])
                end
                #k = first(keys(prand_σ))
                for k in keys(prand_σ)
                    ne = length(prand_σ[k])
                    d = ne == 1 ?
                        fit_mean_Σ(LogNormal, 1, prand_σ[k]) :
                        fit_mean_Σ(MvLogNormal, fill(1, ne), PDiagMat(exp.(prand_σ[k])))
                    indiv_random[k, i_indiv] ~ d
                    #indiv_random[k,i_indiv] = rand(d)
                end
            end
            #poptl = CA.ComponentVector(popt, first(getaxes(popt0)))
            #poptl = CA.ComponentVector(
            #   vcat(fixed, random, prand_σ, vec(indiv), vec(indiv_random)), first(getaxes(popt0)))
            #@show poptl
            #Main.@infiltrate_main
            # sampling changes eltype of Any, need to convert back
            res_sim = sim_sols_probs(convert(typeof(fixed_nl),
                    getdata(fixed))::typeof(fixed_nl),
                convert(typeof(random_nl), getdata(random))::typeof(fixed_nl),
                convert(typeof(indiv_nl), getdata(indiv))::typeof(indiv_nl),
                convert(typeof(indiv_random_nl),
                    getdata(indiv_random))::typeof(indiv_random_nl);
                #saveat = saveat)
            )
            #i_indiv = 1
            for i_indiv in 1:n_indiv
                # not in 1.6 (; sol, problem_opt) = res_sim[i_indiv]
                (sol, problem_opt) = res_sim[i_indiv]
                #!is_dual && @show popt, sol.retcode
                if !SciMLBase.successful_retcode(sol.retcode)
                    Turing.@addlogprob! -Inf
                    return
                end
                local parl = label_par(psets.fixed, problem_opt.p) #@inferred label_par(psetci, problem_opt.p)
                # for accessing solution at 100 time points need to store full solution
                for stream in streams
                    obss = obs[i_indiv][stream]
                    pred = sol(obss.t; idxs = stream_nums[stream]).u
                    #(i,t) = first(enumerate(obss.t))
                    for (i, t) in enumerate(obss.t)
                        pred_t = pred[i]
                        unc = obss.obs_unc[i]
                        dist_pred = fit_mean_Σ(dtypes[stream], pred_t, unc)
                        #tmp = rand(dist_pred)
                        obs_target[i_indiv][stream][i] ~ dist_pred
                        #obss.obs[i] = rand(dist_pred)
                    end
                end
                # TODO allow for specialized adjustment of logprob
            end # for i_indiv
            :return_from_tmodel_cross
        end # function      
    end # let
    #
    # target observations need to be subsequent values -> extract from general obs
    # that holds also the uncertainties and more
    gen_tmodel_cross(obs_target)
end

"""
    get_init_mixedmodel(fixed::ComponentVector, random::ComponentVector, indiv::ComponentMatrix,
        priors_σ, indiv_random)

Construct a ComponentVector corresponding to the parameters sampled by the mixed model.
Argument `indiv` should hold individual identifiers as column axis.

The return has components
`fixed`, `random`, `random_σ`, `indiv`, `indiv_random`
where `indiv` is a flat version with column names as entries of vectors.
"""
function get_init_mixedmodel(psets, popt_indiv::AbstractVector{<:ComponentVector}, priors_σ;
        kwargs...)
    (fixed, random, indiv, indiv_random) = extract_mixed_effects(psets, popt_indiv;
        kwargs...)
    get_init_mixedmodel(; fixed, random, indiv, priors_σ, indiv_random)
end

function extract_mixed_effects(psets, popt_indiv::AbstractVector{<:ComponentVector};
        indiv_ids = ((Symbol("i$i") for i in 1:length(popt_indiv))...,))
    keys_p = map(pset -> keys(axis_paropt_flat1(pset)), psets)
    keys_opt = MTKHelpers.tuplejoin(keys_p...)
    #k = first(keys_opt) # k = keys_opt[2]
    popt_mean = ComponentVector(;
        ((k, mean(getproperty.(popt_indiv, k))) for k in keys_opt)...)
    fixed = popt_mean[keys_p.fixed]
    random = popt_mean[keys_p.random]
    ax_site = Axis(indiv_ids)
    #@show indiv_ids, ax_site
    indiv = ComponentMatrix(hcat((popt[keys_p.indiv] for popt in popt_indiv)...),
        axis_paropt_flat1(psets.indiv), ax_site)
    #popt = first(popt_indiv)
    indiv_random = ComponentMatrix(hcat((popt[keys_p.random] ./ popt_mean[keys_p.random] for popt in popt_indiv)...),
        axis_paropt_flat1(psets.random), ax_site)
    (; fixed, random, indiv, indiv_random)
end

function get_init_mixedmodel(; fixed::ComponentVector, random::ComponentVector,
        indiv::ComponentMatrix, priors_σ,
        indiv_random = missing)
    random_σ = random_σ = ComponentVector(;
        ((k, mean(priors_σ[k])) for k in keys(random))...)
    if ismissing(indiv_random)
        n_indiv = size(indiv, 2)
        ax_random = first(getaxes(random))
        ax_site = getaxes(indiv)[2]
        indiv_random = ComponentMatrix(fill(1.0, length(random), n_indiv),
            ax_random, ax_site)
    end
    ComponentVector(; fixed, random,
        random_σ,
        indiv = flatten_cm(indiv),
        indiv_random = flatten_cm(indiv_random),)
end

"""
    flatten_cm(cm::ComponentMatrix)

Return a flat version of ComponentMatrix cm.
"""
function flatten_cm(cm::ComponentMatrix)
    template = ComponentVector(; ((k, cm[:, k]) for k in keys(getaxes(cm)[2]))...)
    ComponentArray(vec(cm), getaxes(template))
end

# MAYBE - move to proper place
# function map_keys2(FUN, cv::ComponentVector; rewrap::Val{is_rewrap}=Val(true)) where {is_rewrap}
#     f1 = (k) -> begin
#         ret = FUN(k, cv[k])
#         eltype(ret) != Union{} ||
#             error("For mapping empty keys, provide a proper eltype." *
#                   "Did you accidentally write key=[] instead of e.g. Float64[] ?")
#         (k, ret)
#     end
#     if is_rewrap
#         gen = (f1(k) for k in keys(cv))
#         ComponentVector(;gen...)
#     else
#         map(k -> FUN(k, cv[k]), keys(cv))
#     end
# end
