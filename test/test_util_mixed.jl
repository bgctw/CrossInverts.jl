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

toolsA = setup_tools_scenario(:A; scenario, popt, system, random);

psets = setup_psets_fixed_random_indiv(system, popt, keys(fixed), keys(random))

@testset "setup_psets_fixed_random_indiv" begin
    @test all((:fixed, :random, :indiv) .∈ Ref(keys(psets)))
    @test psets.fixed isa ODEProblemParSetter 
    @test get_paropt_labeled(toolsA.pset, toolsA.problem; flatten1=Val(true)) == popt
    @test get_paropt_labeled(psets.fixed, toolsA.problem; flatten1=Val(true)) == fixed
    @test get_paropt_labeled(psets.random, toolsA.problem; flatten1=Val(true)) == random
    @test get_paropt_labeled(psets.indiv, toolsA.problem; flatten1=Val(true)) == indiv
end;


tmpf = () -> begin
    #using DistributionFits
    #using StatsPlots
    problem = toolsA.problem
    priors_random = toolsA.priors_random
    plot(dist); vline!([1.0])
    rand(dist, 2)
end

p_sites = CP.get_site_parameters(Val(:CrossInverts_samplesystem1))
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
tools1 = df.tool[1];

_setuptools = (u0, p) -> setup_tools_scenario(:A; scenario, popt, system, u0, p, random);
_tools = _setuptools(df.u0[1], df.p[1]);
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_setuptools) => :tool)
get_paropt_labeled(tools1.pset, _tools.problem)

@testset "simsols" begin
    pset = df.tool[1].pset
    solver = AutoTsit5(Rodas5P())
    sim_sols_probs = gen_sim_sols_probs(; tools = df.tool, psets, solver)
    sols_probs = sim_sols_probs(fixed, random, hcat(df.indiv...), hcat(df.indiv_random...));
    # recomputed sites ranef and set indiv, but used mean fixed parameters
    probo = first(sols_probs).problem_opt
    popt1 = get_paropt_labeled(pset, df.tool[1].problem; flatten1 = Val(true))
    popt2 = get_paropt_labeled(pset, probo; flatten1 = Val(true))
    @test popt2[keys(random)] == first(popt_sites)[keys(random)]
    @test popt2[keys(indiv)] == first(popt_sites)[keys(indiv)]
    @test popt2[keys(fixed)] == popt_mean[keys(fixed)] # first(popt_sites)[keys(fixed)]
    sol = first(sols_probs).sol;
    @test all(sol[sv.x][1] .== p_sites.A.u0) # state only u0 all randomeffect
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
