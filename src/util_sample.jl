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

        Turing.@model function tmodel_cross(obs, ::Type{T} = Float64) where {T}
            npfix = count_paropt(psets.fixed)
            nprand = count_paropt(psets.random)
            npsite = count_paropt(psets.indiv)
            fixed = StaticArrays.MVector{npfix, T}(undef)
            for (i, r) in enumerate(priors_pop.fixed)
                fixed[i] ~ r
                #popt[i0_fixed+i] ~ r # = popt0[i0_fixed+i]
            end
            #random = Vector{T}(undef, nprand)
            random = StaticArrays.MVector{nprand, T}(undef)
            for (i, r) in enumerate(priors_pop.random)
                random[i] ~ r # = popt0[i0_μb+i] #mean(r)
                #popt[i0_μb+i] ~ r # = popt0[i0_μb+i] #mean(r)
            end
            #prand_σstar = Vector{T}(undef, nprand)
            prand_σstar = StaticArrays.MVector{nprand, T}(undef)
            for (i, r) in enumerate(priors_pop.random_σstar)
                #popt[i0_σb+i] ~r # = popt0[i0_σb+i] #mean(r)
                prand_σstar[i] ~ r # = popt0[i0_σb+i] #mean(r)
            end
            # indiv = Matrix{T}(undef, npsite, n_indiv)
            # indiv_random = Matrix{T}(undef, nprand, n_indiv)
            indiv = StaticArrays.MMatrix{npsite, n_indiv, T}(undef)
            indiv_random = StaticArrays.MMatrix{nprand, n_indiv, T}(undef)
            for i_indiv in 1:n_indiv
                # i0_isite = i0_sites + (i_indiv-1)*count_paropt(psets.indiv)
                for (i, r) in enumerate(priors_indiv[i_indiv])
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
                #saveat = saveat)
                )
            #i_indiv = 1
            for i_indiv in 1:n_indiv
                obsi = obs[i_indiv]
                # not in 1.6 (; sol, problem_opt) = res_sim[i_indiv]
                (sol, problem_opt) = res_sim[i_indiv]
                #!is_dual && @show popt, sol.retcode
                if !SciMLBase.successful_retcode(sol.retcode)
                    Turing.@addlogprob! -Inf
                    return
                end
                local parl = label_par(psets.fixed, problem_opt.p) #@inferred label_par(psetci, problem_opt.p)
                # for accessing solution at 100 time points need to store full solution
                for stream in keys(obs)
                    obss = obsi[stream]
                    pred = sol(obss.t; idxs=stream_nums[stream]).u
                    #(i,t) = first(enumerate(obss.t))
                    for (i,t) in enumerate(obss.t)
                        pred_t = pred[i]
                        unc = obss.obs_unc[i]
                        dist_pred = fit_mean_Σ(dtypes[s], pred_t, unc)
                        #tmp = rand(dist_pred)
                        obss[i] ~ dist_pred
                    end
                end
                # TODO allow for specialized adjustment of logprob
            end # for i_indiv
            :return_from_tmodel_cross
        end # function      
    end # let
    #
    gen_tmodel_cross(obs)
end

