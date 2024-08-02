using CrossInverts
using CrossInverts: CrossInverts as CP
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
using MTKHelpers
using ModelingToolkit, OrdinaryDiffEq 

@named sv = CP.samplesystem_vec()
@named system = embed_system(sv)
scenario = (system = :CrossInverts_samplesystem_vec,)

fixed = CA.ComponentVector(sv₊p = 1:3)
random = CA.ComponentVector(sv₊x = [2.1, 2.2], sv₊τ = 0.1)
indiv = CA.ComponentVector(sv₊i = 0.2)
popt = vcat_statesfirst(fixed, random, indiv; system)

psets = setup_psets_mixed(keys(fixed), keys(random); system, popt)


# fixed = CA.ComponentVector(par=(sv₊p = 1:3,))
# random = CA.ComponentVector(state=(sv₊x = [2.1,2.2],), par=(sv₊τ = 0.1,))
# indiv = CA.ComponentVector(par=(sv₊i = 0.2,))
# popt = merge_subvectors(fixed, random, indiv; mkeys=(:state, :par))

toolsA = setup_tools_indiv(:A; scenario, popt, system);
df_site_u0_p = DataFrame(indiv_id = [:A, :B, :C],
    u0 = fill((get_state_labeled(toolsA.pset, toolsA.problem)), 3),
    p = fill((get_par_label(toolsA.pset, toolsA.problem)), 3))
df = copy(df_site_u0_p)




@testset "gen_compute_indiv_rand" begin
    _compute_indiv_rand = gen_compute_indiv_rand(toolsA.pset, random)
    tmp = _compute_indiv_rand(toolsA.problem.u0, toolsA.problem.p)
    @test keys(tmp) == keys(random)
    tmp[:sv₊x] = popt[:sv₊x] ./ random[:sv₊x] # popt -> toolsA -> df_site.u0/p
end

_compute_indiv_rand = gen_compute_indiv_rand(toolsA.pset, random)
DataFrames.transform!(df,
    [:u0, :p] => DataFrames.ByRow(_compute_indiv_rand) => :indiv_random)

# extract the parameters to optimize that are individual-specific to clumns :indiv
_extract_indiv = (u0, p) -> vcat(u0, p)[symbols_paropt(psets.indiv)]
tmp = _extract_indiv(df.u0[1], df.p[1])
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_extract_indiv) => :indiv)

_setuptools = (u0, p) -> setup_tools_indiv(:A; scenario, popt, system, u0, p);
_setuptools(df.u0[1], df.p[1]);
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_setuptools) => :tool)

@testset "simsols" begin
    solver = AutoTsit5(Rodas5P())
    sim_sols_probs = gen_sim_sols_probs(; tools = df.tool, psets, solver)
    sols_probs = sim_sols_probs(fixed, random, hcat(df.indiv...), hcat(df.indiv_random...))
    sol = first(sols_probs).sol
    solA0 = solve(toolsA.problem, solver) # same as with initial popt -> recomputed
    solA0 == sol
    #sol.t
    #sol[sv.x]

    sim_sols = gen_sim_sols(; tools = df.tool, psets, solver, maxiters = 1e4)
    poptl = (;
        fixed,
        random,
        indiv = hcat(df.indiv...),
        indiv_random = hcat(df.indiv_random...),)
    sols = sim_sols(poptl)
    sol = first(sols)
    solA0 == sol
end;

@testset "sample_slc_cross" begin
    lc = (targetlim = :lim_P,
        scenario = (:prior_sep, :prior_B, :freek_LP, :test_util_optim))
    sites = [:BBR, :LUE]
    n_sample = 2
    #n_sample = 90
    df_site_u0_p = with_logger(MinLevelLogger(current_logger(), Logging.Warn)) do
        #indiv_id = last(sites)
        res_optims = pmap(sites) do indiv_id
            slc = (; indiv_id, lc...)
            #(;u0, p, res_opt) = CP.optim_slc(slc; iterations=2, iterations2=2);
            (; u0, p, res_opt) = CP.optim_slc(slc; iterations = 400, iterations2 = 100)
            (; indiv_id, u0, p)
        end
        df_site_u0_p = DataFrame(res_optims)
    end
    sd_df = subset(CP.get_sitedata_df(), :layer => ByRow(==(:top30)))
    fsitedata = (indiv_id) -> sd_df[findfirst(==(indiv_id), sd_df.indiv_id), :]
    transform!(df_site_u0_p, :indiv_id => ByRow(fsitedata) => :sitedata)
    tools1 = with_logger(MinLevelLogger(current_logger(), Logging.Warn)) do
        tools1 = setup_tools_indiv(sites[1], lc...; sitedata = df_site_u0_p.sitedata[1])
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
              Ref(names(chn, :random_σ)))
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
    # TODO think about labelling solution 
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
    describe(MCMCChains.Chains(chn, [:random_σ]))
    chn2 = MCMCChains.Chains(chn, [:random_σ])
    describe(MCMCChains.get(chn, :lp).lp)
end
