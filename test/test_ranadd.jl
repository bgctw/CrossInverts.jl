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

(; indivdata, p_indiv, d_noise) = CP.simulate_case_indivdata(inv_case; scenario)

@testset "initial values of random effects" begin
    @test pop_info.mixed_keys ==
          (fixed = (), ranadd = (:b1, :b2), ranmul = (), indiv = ())
    # setup_inversion indiv_info differs from p_indiv: par_opt (and corresponding p and u0 set to mean)
    # initialized all individuals to mean random effect
    @test all(pop_info.mixed.indiv_ranadd .== 0.0)
    p_indiv.indiv_ranadd
    @test p_indiv.p[1][:b3] == 0.0 # check b3 zero
    x1 = p_indiv.u0[1].x1
    b1s = map(p -> p.b1, p_indiv.p)
    b2s = map(p -> p.b2, p_indiv.p)
    #
    # test that simulation with true parameters yields same as generated observations
    mixed_true = CP.extract_mixed_effects(pop_info.psets, p_indiv.paropt)
    sims_true = sols_probs = sim_sols_probs(mixed_true...)
    sol1 = sims_true[1].sol
    symdict = get_system_symbol_dict(system)
    y1 = sol1[Symbolics.scalarize(symdict[:y])]
    @test all(isapprox.(indivdata.i1.y.obs_true, y1))
end;

tmpf = () -> begin
    n_burnin = 50
    n_thread = 4
    n_sample = trunc(Int, 800 / n_thread)
end;

# test no warnings on empty effects sets indiv, ranmul, ...
n_burnin = 0
n_thread = 2
n_sample = 10

model_cross = gen_model_cross(;
    inv_case, tools = indiv_info.tools,
    priors_pop = pop_info.priors_pop,
    psets = pop_info.psets,
    sim_sols_probs, scenario, solver);

chn = Turing.sample(model_cross, Turing.NUTS(n_burnin, 0.65, init_ϵ = 0.025),
    MCMCThreads(), n_sample, n_thread, init_params = collect(pop_info.sample0))

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
    chn2 = chn[:, vcat(pop_info.effect_pos[:indiv_ranadd][:i2]...), :]
    @test names(chn2) == [Symbol("indiv_ranadd[:b1, 2]"),
        Symbol("indiv_ranadd[:b2, 2]")]
end;

@testset "random effects" begin
    chn_r = extract_group(chn, :ranadd, pop_info.indiv_ids)
    #SP.plot(chn_r)  # should converge towards 1 and 2 with more samples
    chn_ri = extract_group(chn, :indiv_ranadd)
    chn_ri1 = extract_group(chn_ri, Symbol(":b1"))
    if prod(size(chn, [1, 3])) >= 400
        b1m = transpose(mapslices(mean, Array(chn_r); dims = 1))
        @test all(isapprox.(b1m, [1.0, 2.0]; rtol = 0.3))
        # samples of confidence interval encompass true value of individual effects
        b1im = transpose(mapslices(mean, Array(chn_ri1); dims = 1))
        b1isd = transpose(mapslices(std, Array(chn_ri1); dims = 1))
        b1i_true = map(p -> p.b1, p_indiv.indiv_ranadd)
        #SP.density(chn_ri1)  # individual offsets for b1
        i = 1
        for in in 1:length(b1i_true)
            @test b1i_true[i] >= b1im[i] - 2 * b1isd[i]
            @test b1i_true[i] <= b1im[i] + 2 * b1isd[i]
        end
    end
    #chn_ri1 =  extract_group(chn_ri, Symbol(":b2"))
    #SP.density(b1m) # distributed normally fine
    # with more samples should converge towards     
end;

@testset "test empty ranadd" begin
    scenario_e = (:all_ranmul,)
    #res_e = setup_inversion(inv_case; scenario = scenario_e);
    res_s = CP.simulate_case_indivdata(inv_case; scenario = scenario_e)
    @test isempty(res_s.p_indiv.indiv_ranadd[1])
    @test !isempty(res_s.p_indiv.indiv_ranmul[1])
end
