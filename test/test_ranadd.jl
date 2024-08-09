using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using ModelingToolkit, OrdinaryDiffEq
using MTKHelpers
using DataFrames
using ComponentArrays: ComponentArrays as CA
using StableRNGs
using Statistics
using DistributionFits
using PDMats: PDiagMat
using Turing

inv_case = SampleSystemStaticCase()
scenario = (:all_ranadd,)

(; system, indiv_info, pop_info) = setup_inversion(inv_case; scenario)

solver = AutoTsit5(Rodas5P())
sim_sols_probs = gen_sim_sols_probs(;
    tools = indiv_info.tools, psets = pop_info.psets,
    problemupdater = pop_info.problemupdater, solver)

tmpf = () -> begin
    (; fixed, ranadd, ranmul, indiv, indiv_ranadd, indiv_ranmul) = pop_info.mixed
    (; psets, problemupdater, priors_pop, sample0, effect_pos) = pop_info
    mixed_keys = pop_info.mixed_keys
    sols_probs = sim_sols_probs(pop_info.mixed...)
    n_burnin = 50
    n_thread = 4
    n_sample = 800/n_thread
  end;

n_burnin = 0
n_thread = 2
n_sample = 10

model_cross = gen_model_cross(;
    inv_case, tools = indiv_info.tools,
    priors_pop = pop_info.priors_pop,
    psets = pop_info.psets,
    sim_sols_probs, scenario, solver);

chn = Turing.sample(model_cross, Turing.NUTS(n_burnin, 0.65, init_ϵ = 0.025),
    MCMCThreads(), n_sample, 2, init_params = collect(pop_info.sample0))

tmpf = () -> begin
    # for finding initial step size use some more adaptive steps
    chn = Turing.sample(model_cross, Turing.NUTS(1000, 0.65), n_sample,
        init_params = collect(pop_info.sample0))
    names(chn, :parameters)
    # first chain as a ComponentMatrix
    s1 = CA.ComponentMatrix(Array(chn[:, 1:length(sample0), 1]),
        CA.FlatAxis(), first(CA.getaxes(sample0)))
    s1[:, :fixed][:, :sv₊p]
end

@testset "Turing indices match sample" begin
    chn2 = chn[:, vcat(effect_pos[:indiv_ranmul][:B]...), :]
    @test names(chn2) == [Symbol("indiv_ranmul[:sv₊x, 2][1]"),
        Symbol("indiv_ranmul[:sv₊x, 2][2]"), Symbol("indiv_ranmul[:sv₊τ, 2]")]
end;
