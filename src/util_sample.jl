function gen_model_cross(;
        tools, priors_pop, sim_sols_probs, scenario, psets, solver)
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
        dtypes = (;zip(streams, (get_obs_uncertainty_dist_type(Val(scenario.system), s; scenario) for s in streams))...)
        #saveat = union(map_keys(stream -> stream.t, obs)),
        stream_nums = (;
            zip(streams,
                Symbolics.scalarize.(getindex.(Ref(sys_num_dict), streams)))...)

        Turing.@model function tmodel_cross(obs_target, ::Type{T} = Float64) where {T}
            npfix = count_paropt(psets.fixed)
            nprand = count_paropt(psets.random)
            npsite = count_paropt(psets.indiv)
            fixed = StaticArrays.MVector{npfix, T}(undef)
            fixed_l = label_paropt_flat1(psets.fixed, fixed)
            k = first(keys(fixed_l))
            for k in keys(fixed_l)
                fixed_l[k] ~ priors_pop.fixed[k]
                #fixed_l[k] = rand(priors_pop.fixed[k])
            end
            #random = Vector{T}(undef, nprand)
            random = StaticArrays.MVector{nprand, T}(undef)
            random_l = label_paropt_flat1(psets.random, random)
            #k = first(keys(random_l))
            for k in keys(random_l)
                random_l[k] ~ priors_pop.random[k]
                #random_l[k] = rand(priors_pop.random[k])
            end
            #prand_σ = Vector{T}(undef, nprand)
            prand_σ = StaticArrays.MVector{nprand, T}(undef)
            prand_σ_l = label_paropt_flat1(psets.random, random)
            #k = first(keys(prand_σ_l))
            for k in keys(prand_σ_l)
                prand_σ_l[k] ~ priors_pop.random_σ[k]
                #prand_σ_l[k] = rand(priors_pop.random_σ[k])
            end
            # indiv = Matrix{T}(undef, npsite, n_indiv)
            # indiv_random = Matrix{T}(undef, nprand, n_indiv)
            #indiv1_l = label_paropt_flat1(psets.indiv, 1:count_par(psets.indiv))
            #template = label_paropt_flat1(psets.indiv, 1:count_paropt(psets.indiv))
            ax_indiv = axis_paropt_flat1(psets.indiv)
            indiv = StaticArrays.MMatrix{npsite, n_indiv, T}(undef)
            indiv_l = ComponentArray(getdata(indiv), ax_indiv, FlatAxis())
            indiv_random = StaticArrays.MMatrix{nprand, n_indiv, T}(undef)
            indiv_random_l = ComponentArray(getdata(indiv_random), first(getaxes(random_l)), FlatAxis())
            #i_indiv = 1
            for i_indiv in 1:n_indiv
                prior_indiv = priors_indiv[i_indiv]
                #k = first(keys(prior_indiv))
                for k in keys(prior_indiv)
                    indiv_l[k,i_indiv] ~ prior_indiv[k]
                    #indiv_l[k,i_indiv] = rand(prior_indiv[k])
                end
                #k = first(keys(prand_σ_l))
                for k in keys(prand_σ_l)
                    ne = length(prand_σ_l[k])
                    d = ne == 1 ? 
                    fit_mean_Σ(LogNormal, 1, prand_σ_l[k]) :
                    fit_mean_Σ(MvLogNormal, fill(1,ne), PDiagMat(exp.(prand_σ_l[k])))
                    indiv_random_l[k,i_indiv] ~ d
                    #indiv_random_l[k,i_indiv] = rand(d)
                end
            end
            #poptl = CA.ComponentVector(popt, first(getaxes(popt0)))
            #poptl = CA.ComponentVector(
            #   vcat(fixed, random, prand_σ, vec(indiv), vec(indiv_random)), first(getaxes(popt0)))
            #@show poptl
            #Main.@infiltrate_main
            res_sim = sim_sols_probs(getdata(fixed_l),
                getdata(random_l),
                getdata(indiv_l),
                getdata(indiv_random_l);
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
                    pred = sol(obss.t; idxs=stream_nums[stream]).u
                    #(i,t) = first(enumerate(obss.t))
                    for (i,t) in enumerate(obss.t)
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

