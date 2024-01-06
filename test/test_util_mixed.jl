using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using ModelingToolkit, OrdinaryDiffEq
using MTKHelpers
using DataFrames
using ComponentArrays: ComponentArrays as CA

# df_site_u0_p = DataFrames.DataFrame([
#         (:siteA,
#             CA.ComponentVector(mv₊x = [2.1,2.2]),
#             CA.ComponentVector(mv₊p1 = 10, mv₊p2 = 5.2, mv₊τ = 0.1)),
#         (:siteB,
#             CA.ComponentVector(mv₊x = 1.8),
#             CA.ComponentVector(mv₊p1 = 11, mv₊p2 = 4.8, mv₊τ = 0.1)),
#     ], [:site, :u0, :p])

# u0 = df_site_u0_p.u0[1]
# p = df_site_u0_p.p[1]
@named mv = CP.samplesystem_vec()
@named system = embed_system(mv)
scenario = (system = :CrossInverts_samplesystem_vec,)

popt = CA.ComponentVector(mv₊x = [3.1, 3.2], mv₊τ = 0.5, mv₊p = 1:3)

u0 = CA.ComponentVector(mv₊x = [3.11, 3.21])
p = CA.ComponentVector(mv₊τ = 0.51)

tools = setup_tools_scenario(:A, scenario, popt; system);
probo = remake(tools.problem, popt, tools.pset)

fixed = CA.ComponentVector(mv₊p = 1:3)
random = CA.ComponentVector(mv₊x = [2.1,2.2], mv₊τ = 0.1)
keys_indiv = (:mv₊i,)
keys_popt = (keys(fixed)..., keys(random)..., keys_indiv...)

psets = setup_psets_fixed_random_indiv(get_system(tools.problem), keys_popt,
    keys(fixed), keys(random))


df_site_u0_p = DataFrame(
    site = [:A, :B, :C],
    u0 = fill((label_state(tools.pset, tools.problem.u0)), 3),  
    p = fill((label_par(tools.pset, tools.problem.p)), 3),   
)
df = copy(df_site_u0_p)
# compute indiv_random by indiv/random to columns :indiv_random
_compute_indiv_rand = (u0, p) -> begin
    local popt = get_paropt_labeled(tools.pset, u0, p; flatten1=Val(true))
    # k = first(keys(random))
    gen = (popt[k] ./ random[k] for k in keys(random))
    v = reduce(vcat, gen)
    MTKHelpers.attach_axis(v, first(CA.getaxes(random)))
end
tmp = _compute_indiv_rand(df.u0[1], df.p[1])
DataFrames.transform!(df,
    [:u0, :p] => DataFrames.ByRow(_compute_indiv_rand) => :indiv_random)

# extract the parameters to optimize that are individual-specific to clumns :indiv
_extract_indiv = (u0, p) -> vcat(u0, p)[symbols_paropt(psets.indiv)]
tmp = _extract_indiv(df.u0[1], df.p[1])
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_extract_indiv) => :indiv)

_setuptools = (u0, p) -> setup_tools_scenario(:A, scenario, popt; system, u0, p);
_setuptools(df.u0[1], df.p[1]);
DataFrames.transform!(df, [:u0, :p] => DataFrames.ByRow(_setuptools) => :tool)

sim_sols_probs = gen_sim_sols_probs(; tools = df.tool, psets)
sols_probs = sim_sols_probs(fixed, random, hcat(df.indiv...), hcat(df.indiv_random...));
sol = first(sols_probs).sol;
sol.t
sol[mv.x]

sim_sols = gen_sim_sols(; tools = df.tool, psets, maxiters = 1e4)
poptl = (;
    fixed,
    random,
    indiv = hcat(df.indiv...),
    indiv_random = hcat(df.indiv_random...),
    )
sols = sim_sols(poptl);
