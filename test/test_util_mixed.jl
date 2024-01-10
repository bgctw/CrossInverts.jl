using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using ModelingToolkit, OrdinaryDiffEq
using MTKHelpers
using DataFrames
using ComponentArrays: ComponentArrays as CA
using StableRNGs
using Statistics

@named sv = CP.samplesystem_vec()
@named system = embed_system(sv)
scenario = (system = :CrossInverts_samplesystem_vec,)

fixed = CA.ComponentVector(sv₊p = 1:3)
random = CA.ComponentVector(sv₊x = [2.1,2.2], sv₊τ = 0.1)
indiv = CA.ComponentVector(sv₊i = 0.2)
popt = vcat_statesfirst(fixed, random, indiv; system)

# fixed = CA.ComponentVector(par=(sv₊p = 1:3,))
# random = CA.ComponentVector(state=(sv₊x = [2.1,2.2],), par=(sv₊τ = 0.1,))
# indiv = CA.ComponentVector(par=(sv₊i = 0.2,))
# popt = merge_subvectors(fixed, random, indiv; mkeys=(:state, :par))

toolsA = setup_tools_scenario(:A; scenario, popt, system, keys_indiv = keys(indiv));

psets = setup_psets_fixed_random_indiv(keys(fixed), keys(random); system, popt)

@testset "setup_psets_fixed_random_indiv" begin
    @test all((:fixed, :random, :indiv) .∈ Ref(keys(psets)))
    @test psets.fixed isa ODEProblemParSetter 
    @test get_paropt_labeled(toolsA.pset, toolsA.problem; flatten1=Val(true)) == popt
    @test get_paropt_labeled(psets.fixed, toolsA.problem; flatten1=Val(true)) == fixed
    @test get_paropt_labeled(psets.random, toolsA.problem; flatten1=Val(true)) == random
    @test get_paropt_labeled(psets.indiv, toolsA.problem; flatten1=Val(true)) == indiv
end;

priors_pop = setup_priors_pop(keys(fixed), keys(random); scenario);

tmpf = () -> begin
    #using DistributionFits
    #using StatsPlots
    problem = toolsA.problem
    priors_random = toolsA.priors_random
    plot(dist); vline!([1.0])
    rand(dist, 2)
end

p_sites = CP.get_site_parameters(Val(scenario.system))
df = DataFrame(
    site = collect(keys(p_sites)), 
    u0 = collect(CP.map_keys(x -> x.u0, p_sites; rewrap = Val(false))),
    p = collect(CP.map_keys(x -> x.p, p_sites; rewrap = Val(false))),
)

popt_sites = get_paropt_labeled.(Ref(toolsA.pset), df.u0, df.p; flatten1=Val(true))
#getproperty.(popt_sites, :sv₊p)
_tup = map(k -> mean(getproperty.(popt_sites, k)), keys(popt)) 
popt = popt_mean = CA.ComponentVector(;zip(keys(popt), _tup)...)
fixed = popt_mean[keys(fixed)]
random = popt_mean[keys(random)]


@testset "gen_compute_indiv_rand" begin
    _compute_indiv_rand = gen_compute_indiv_rand(toolsA.pset, random)    
    tmp = _compute_indiv_rand(toolsA.problem.u0, toolsA.problem.p)
    @test keys(tmp) == keys(random)
    tmp[:sv₊x] = popt_mean[:sv₊x] ./ random[:sv₊x] # popt -> toolsA -> df_site.u0/p
end

_compute_indiv_rand = gen_compute_indiv_rand(toolsA.pset, random)    
DataFrames.transform!(df,
[:u0, :p] => DataFrames.ByRow(_compute_indiv_rand) => :indiv_random)

# extract the parameters to optimize that are individual-specific to clumns :indiv
_extract_indiv = (u0, p) -> vcat(u0, p)[symbols_paropt(psets.indiv)]
tmp = _extract_indiv(df.u0[1], df.p[1])
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_extract_indiv) => :indiv)

_setuptools = (u0, p) -> setup_tools_scenario(:A; scenario, popt, system, u0, p, 
    keys_indiv = keys(indiv));
_tools = _setuptools(df.u0[1], df.p[1]);
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_setuptools) => :tool)

tools1 = df.tool[1]; 
tools = df.tool
get_paropt_labeled(tools1.pset, _tools.problem)

@testset "simsols" begin
    pset = df.tool[1].pset
    solver = AutoTsit5(Rodas5P())
    sim_sols_probs = gen_sim_sols_probs(; tools = df.tool, psets, solver)
    sols_probs = sim_sols_probs(fixed, random, hcat(df.indiv...), hcat(df.indiv_random...);
        saveat = [0,0.2,2]);
    (sol, problem_opt) = sols_probs[1];
    # recomputed sites ranef and set indiv, but used mean fixed parameters
    popt1 = get_paropt_labeled(pset, df.tool[1].problem; flatten1 = Val(true))
    popt2 = get_paropt_labeled(pset, problem_opt; flatten1 = Val(true))
    @test popt2[keys(random)] == first(popt_sites)[keys(random)]
    @test popt2[keys(indiv)] == first(popt_sites)[keys(indiv)]
    @test popt2[keys(fixed)] == popt_mean[keys(fixed)] # first(popt_sites)[keys(fixed)]
    sol = first(sols_probs).sol;
    @test all(sol[sv.x][1] .== p_sites.A.u0) # state only u0 all randomeffect
    sol[sv.x]
    sol([0.3,0.35]; idxs=[sv.dec2, sv.dec2])
    @test all(sol([0.3,0.35]; idxs=sv.dec2).u .> 0) # observed at a interpolated times
    solA0 = solve(df.tool[1].problem, solver) 
    #solA0 == sol # random and individual parameters for original problem at mean instead of site

    sim_sols = gen_sim_sols(; tools = df.tool, psets, solver, maxiters = 1e4)
    poptl = (;
        fixed,
        random,
        indiv = hcat(df.indiv...),
        indiv_random = hcat(df.indiv_random...),
        )
    sols = sim_sols(poptl);
    sol2 = first(sols);
    @test sol2 == sol
end;

@testset "extract_stream_obsmatrices" begin
    tools = df.tool
    vars = (:obs, :obs_true)
    obs = CP.extract_stream_obsmatrices(;tools, vars)
    @test keys(obs) == (:sv₊x, :sv₊dec2) # streams
    @test all( (keys(obs[k]) == (:t, vars...) for k in keys(obs)) ) # keys in each stream
    #
    # inconsistent times
    tools2 = deepcopy(tools)
    tools2[1].sitedata.sv₊x.t[1] = 0.222
    @test_throws (ErrorException) CP.extract_stream_obsmatrices(;tools = tools2, vars)
    #
    # inconsistent length of observations
    tools2 = deepcopy(tools)
    tools2[1] = merge(tools2[1],
        (;
            sitedata = merge(tools2[1].sitedata,
                (;
                    sv₊x = merge(tools2[1].sitedata.sv₊x,
                        (; obs = tools2[1].sitedata.sv₊x.obs[2:end]))))))
    @test_throws (ErrorException) CP.extract_stream_obsmatrices(; tools = tools2, vars)
    #
    # consistent observations but inconsistent with time
    tools2 = deepcopy(tools)
    tools2_i = tools2[1]
    tools2 = [merge(tools2_i,
        (;
            sitedata = merge(tools2_i.sitedata,
                (;
                    sv₊x = merge(tools2_i.sitedata.sv₊x,
                        (; obs = tools2_i.sitedata.sv₊x.obs[2:end]))))))
              for tools2_i in tools2]
    #[t.sitedata.sv₊x.obs for t in tools2] # just to check that each site has only 3 obs
    @test_throws (ErrorException) CP.extract_stream_obsmatrices(; tools = tools2, vars)
end

obs = CP.extract_stream_obsmatrices(;tools)
stream = first(obs)
model_cross = gen_model_cross(;
    tools=df.tool, priors_pop, psets, sim_sols_probs, scenario, solver)
#tmp = model_cross(obs)
