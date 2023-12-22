using SesamFitSPP
using SesamFitSPP: SesamFitSPP as CP
using ComponentArrays: ComponentArrays as CA
using Distributions
using MTKHelpers
using StatsBase
using DataFrames
using Distributed #pmap
using MCMCChains
using Test
using Logging, LoggingExtras
#using Infiltrator
#using MTKHelpers

@testset "sample_slc_cross" begin
    lc = (targetlim = :lim_P,
        scenario = (:prior_sep, :prior_B, :freek_LP, :test_util_optim))
    sites = [:BBR, :LUE]
    n_sample = 2
    #n_sample = 90
    df_site_u0_p = with_logger(MinLevelLogger(current_logger(), Logging.Warn)) do
        #site = last(sites)
        res_optims = pmap(sites) do site
            slc = (; site, lc...)
            #(;u0, p, res_opt) = CP.optim_slc(slc; iterations=2, iterations2=2);
            (; u0, p, res_opt) = CP.optim_slc(slc; iterations = 400, iterations2 = 100)
            (; site, u0, p)
        end
        df_site_u0_p = DataFrame(res_optims)
    end
    sd_df = subset(CP.get_sitedata_df(), :layer => ByRow(==(:top30)))
    fsitedata = (site) -> sd_df[findfirst(==(site), sd_df.site), :]
    transform!(df_site_u0_p, :site => ByRow(fsitedata) => :sitedata)
    tools1 = with_logger(MinLevelLogger(current_logger(), Logging.Warn)) do
        tools1 = setup_tools_scenario(sites[1], lc...; sitedata = df_site_u0_p.sitedata[1])
    end
    #
    keys_opt_fixed = (:s₊a_E, :s₊k_mN_L, :s₊ϵ, :s₊ϵ_tvr)
    keys_opt_random = (:s₊k_L, :s₊k_R, :s₊k_LP)
    #tmp = CP.get_priors_σ_dict(;s₊k_L=Exponential(0.5))
    res = with_logger(MinLevelLogger(current_logger(), Logging.Warn)) do
        res = CP.sample_slc_cross(df_site_u0_p, lc...; keys_opt_fixed, keys_opt_random,
            n_sample, n_burnin = 0)
    end
    chn = res.chn
    #Serialization.serialize("tmp/mixed_sample_chn.js", chn)
    @test all(sort(names(chn, :fixed)) .== sort(collect(keys_opt_fixed)))
    @test all(sort(names(chn, :random)) .== sort(collect(keys_opt_random)))
    @test all([Symbol("σstar_$p") for p in keys_opt_random] .∈
              Ref(names(chn, :random_σstar)))
    par_names = symbols_paropt(tools1.psetci)
    psite_names = setdiff(par_names, union(keys_opt_fixed, keys_opt_random))
    nsite = length(sites)
    @test all([Symbol("s$(nsite)_$p") for p in psite_names] .∈ Ref(names(chn, :indiv)))
    @test all([Symbol("r$(nsite)_$p") for p in keys_opt_random] .∈
              Ref(names(chn, :indiv_random)))
    @test all([Symbol("u0$(nsite)_$p") for p in symbols_state(tools1.psetci)] .∈
              Ref(names(chn, :u0)))

    # extract and label relevant section from sample for a single sample
    s1 = vec(Array(Chains(chn, (res.paropt_sections..., :u0))[1, :, 1]))
    s1l = MTKHelpers.attach_axis(s1, CA.getaxes(res.poptul)[1])
    # simulate
    fsim = CP.gen_sim_sols(; tools = res.tools, psets = res.psets)
    sols = fsim(s1l.popt)
    # test simulate with before updating (non-optimized) starting conditions or parameters
    #   but better provide a updated tools.prob to CP.gen_sim_sols
    pset_u0 = get_concrete(ODEProblemParSetter(tools1.system, symbols_state(tools1.psetci)))
    sols = fsim(s1l.popt; pset_u0, u0 = s1l.u)
    pset_p = get_concrete(ODEProblemParSetter(tools1.system, symbols_par(tools1.psetci)))

    chn2 = set_section(chn,
        merge(parkeys_dict_p_flat, Dict(:internals => names(chn, :internals))))

    namesingroup(chn, :parameters)

    tmpsols = sim_sites(popt0)
    tmpsols[1][end]
    uend = map(sol -> label_state(psets.fixed, sol[:, end]), tmpsols)
    chnu0 = Chains(chn, :u0)

    u0l = ComponentArray(reshape(Array(Chains(chn, :u0)[1, :, 1]), size(parkeys_dict[:u0])),
        (axis_state(psets.fixed), FlatAxis()))
    poptul = ComponentArray(popt = popt0, u = u0l)

    tmp2 = CA.ComponentArray(tmp, getaxes(poptul))

    chn_popt = MCMCChains.Chains(chn, res.paropt_sections)
    chn_u0 = MCMCChains.Chains(chn, :u0)
    p = Array(chn_popt)[1, :]
    nsite = size(res.parkeys_dict[:u0], 2)
    tmp = reshape(Array(chn_u0[1, :, 1]), size(res.parkeys_dict[:u0]))
    tmpv1 = CA.ComponentVector(tmp[:, 1], axis_state(res.psets.fixed))
    tmpv2 = CA.ComponentVector(tmp[:, 2], axis_state(res.psets.fixed))
    tmpm = hcat(tmpv1, tmpv2)
    tmpm2 = CA.ComponentArray(a = tmpm, b = tmpm)
    tmpm3 = CA.ComponentArray(fixed = popt_fixed, random = popt_random)
    sum(popt_fixed_sites; dims = 2)

    tmpl = CA.ComponentArray(tmp, (axis_state(res.psets.fixed), CA.FlatAxis()))

    # tmp = mapslices(Array(chn2); dims=2) do x
    #     #@infiltrate
    #     poptl = CA.ComponentArray(x, CA.getaxes(res.popt0))
    #     sols = res.sim_sites(poptl)
    # end
    # tmp = res.sim_sites(chn[])
end;

tmpf = () -> begin
    describe(MCMCChains.Chains(chn, [:random_σstar]))
    chn2 = MCMCChains.Chains(chn, [:random_σstar])
    describe(MCMCChains.get(chn, :lp).lp)
end
