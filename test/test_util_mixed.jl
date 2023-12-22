using Test
using SesamFitSPP
using SesamFitSPP: SesamFitSPP as CP
using ModelingToolkit, OrdinaryDiffEq
using DataFrames
using ComponentArrays: ComponentArrays as CA

#include("test/samplesystem.jl")
include("samplesystem.jl")

df_site_u0_p = DataFrames.DataFrame([
        (:siteA,
            CA.ComponentVector(d₊x = 2.2),
            CA.ComponentVector(d₊p1 = 10, d₊p2 = 5.2, d₊τ = 0.1)),
        (:siteB,
            CA.ComponentVector(d₊x = 1.8),
            CA.ComponentVector(d₊p1 = 11, d₊p2 = 4.8, d₊τ = 0.1)),
    ], [:site, :u0, :p])

u0 = df_site_u0_p.u0[1]
p = df_site_u0_p.p[1]
@named d = samplesystem()
@named system = embed_system(d)

popt = CA.ComponentVector(d₊x = 3.0, d₊τ = 0.5)

tools = setup_mtk_tools(system, u0, p, keys(popt));
probo = remake(tools.problem, popt, tools.pset)
@test label_state(tools.pset, probo.u0).d₊x == popt.d₊x
@test label_par(tools.pset, probo.p).d₊τ == popt.d₊τ
@test label_par(tools.pset, probo.p).d₊p1 == p.d₊p1

fixed = CA.ComponentVector(d₊p2 = 2.0)
random = CA.ComponentVector(d₊x = 2.0, d₊τ = 0.1)
keys_indiv = (:d₊p1,)
keys_popt = (keys(fixed)..., keys(random)..., keys_indiv...)

psets = setup_psets_fixed_random_indiv(get_system(tools.problem), keys_popt, keys(fixed),
    keys(random))

df = copy(df_site_u0_p)
# compute indiv_random by indiv/random to columns :indiv_random
_compute_indiv_rand = (u0, p) -> begin
    u0p = vcat(u0, p)
    tup = [u0p[k] / random[k] for k in keys(random)]
    CA.ComponentArray(tup, CA.getaxes(random)[1])
end
tmp = _compute_indiv_rand(df.u0[1], df.p[1])
DataFrames.transform!(df,
    [:u0, :p] => DataFrames.ByRow(_compute_indiv_rand) => :indiv_random)

# extract the parameters to optimize that are individual-specific to clumns :indiv
_extract_indiv = (u0, p) -> vcat(u0, p)[symbols_paropt(psets.indiv)]
tmp = _extract_indiv(df.u0[1], df.p[1])
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_extract_indiv) => :indiv)

_setuptools = (u0, p) -> setup_mtk_tools(system, u0, p, keys(popt))
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_setuptools) => :tool)

sim_sols_probs = gen_sim_sols_probs(; tools = df.tool, psets)
sols_probs = sim_sols_probs(fixed, random, hcat(df.indiv...), hcat(df.indiv_random...));

sim_sols = gen_sim_sols(; tools = df.tool, psets)
poptl = (;
    fixed,
    random,
    indiv = hcat(df.indiv...),
    indiv_random = hcat(df.indiv_random...))
sols = sim_sols(poptl);
