struct SampleSystemStaticCase <: AbstractCrossInversionCase end

function samplesystem_static(; name, a = 1.0, b1 = 1.0, b2 = 2.0, b3 = 3.0)
    n_comp = 3
    n_rec = 12
    @parameters t
    D = Differential(t)
    @variables x1(..)[1:n_rec] x2(..)[1:n_rec] x3(..)[1:n_rec] y(..)[1:n_rec]
    ps = @parameters a=a b1=b1 b2=b2 b3=b3
    sts = vcat(
        [x1(t)[r] for r in 1:n_rec],
        [x2(t)[r] for r in 1:n_rec],
        [x3(t)[r] for r in 1:n_rec],
        [y(t)[r] for r in 1:n_rec])
    eq = vcat(
        # keep states constant at initial state
        [D(x1(t)[r]) ~ 0 for r in 1:n_rec],
        [D(x2(t)[r]) ~ 0 for r in 1:n_rec],
        [D(x3(t)[r]) ~ 0 for r in 1:n_rec],
        # simulate regression
        [y(t)[r] ~ a + b1 * x1(t)[r] + b2 * x2(t)[r] + b3 * x3(t)[r] for r in 1:n_rec])
    sys = ODESystem(eq, t, sts, vcat(ps...); name)
    return sys
end

# difficult to specify defaults or ProblemParSetter for Matrix symbolic arrays
# function samplesystem_static(; name, x = reshape(collect(2:37) .* 1.0, 12, 3), a = 1.0, b1 = 1.0, b2 = 1.0, b3 = 1.0)
#     n_comp = 3
#     n_rec = 12
#     @parameters t
#     D = Differential(t)
#     @variables x(..)[1:n_rec, 1:n_comp] y(..)[1:n_rec]
#     ps = @parameters a=a b1=b1 b2=b2 b3=b3
#     sts = vcat(
#         vec([x(t)[r,i] for i in 1:n_comp, r in 1:n_rec]), 
#         [y(t)[r] for r in 1:n_rec])
#     eq = vcat(
#         # keep states constant at initial state
#         vec([D(x(t)[r,i]) ~ 0 for i in 1:n_comp, r in 1:n_rec]), 
#         # simulate regression
#         [y(t)[r] ~ a + b1 * x(t)[r,1] + b2 * x(t)[r,2] + b3 * x(t)[r,3] for r in 1:n_rec])
#     sys = ODESystem(eq, t, sts, vcat(ps...); name)
#     return sys
# end

function get_case_inverted_system(::SampleSystemStaticCase; scenario)
    @named sys_static = samplesystem_static()
    @named system = structural_simplify(sys_static)
    rng = StableRNG(123)
    d_uniform = Uniform(0.0, 5.0)
    u0_default = ComponentVector(
        x1 = rand(rng, d_uniform, 12), x2 = rand(rng, d_uniform, 12), x3 = rand(
            rng, d_uniform, 12))
    p_default = CA.ComponentVector{Float64}(a = 1.0)
    if :all_ranadd ∈ scenario
        p_default = CA.ComponentVector(p_default, b3 = 0.0)
    end
    (; system, u0_default, p_default)
end

function get_case_indiv_ids(::SampleSystemStaticCase; scenario)
    n_indiv = 3
    if :mamy_indiv ∈ scenario
        n_indiv = 12
    end
    Symbol.("i" .* string.(1:n_indiv))
end

function get_case_mixed_keys(::SampleSystemStaticCase; scenario)
    mixed_keys = (;
        fixed = (:a,),
        ranadd = (:b1,),
        ranmul = (:b2,),
        indiv = (:b3,))
    if :all_ranadd ∈ scenario
        mixed_keys = (;
            fixed = (),
            ranadd = (:b1, :b2),
            ranmul = (),
            indiv = ())
    end
    if :single_ranadd ∈ scenario
        mixed_keys = (;
            fixed = (),
            ranadd = (:b1,),
            ranmul = (),
            indiv = ())
    end
    if :all_ranmul ∈ scenario
        mixed_keys = (;
            fixed = (),
            ranadd = (),
            ranmul = (:b1, :b2),
            indiv = ())
    end
    mixed_keys
end

function get_case_priors_dict(
        ::SampleSystemStaticCase, indiv_id; scenario = NTuple{0, Symbol}())
    dn = Normal(0.0, 1.0)
    dln = fit(LogNormal, 1.0, @qp_uu(5.0))
    dd = Dict{Symbol, Distribution}(:a => dn, :b1 => dn, :b2 => dln, :b3 => dn)
    if :all_ranadd ∈ scenario
        dd = Dict{Symbol, Distribution}(
            :b1 => fit(Normal, 1.0, @qp_uu(5.0)),
            :b2 => fit(Normal, 2.0, @qp_uu(5.0))
        )
    end
    if :single_ranadd ∈ scenario    
        dd = Dict{Symbol, Distribution}(
            :b1 => fit(Normal, 1.0, @qp_uu(5.0)),
        )
    end
    if :all_ranmul ∈ scenario
        #dd = Dict{Symbol, Distribution}([:b1, :b2, :b3] .=> dln)
        dd = Dict{Symbol, Distribution}(
            :b1 => fit(LogNormal, 1.0, @qp_uu(5.0), Val(:mode)),
            :b2 => fit(LogNormal, 2.0, @qp_uu(5.0), Val(:mode))
        )
    end
    dd
end

function get_case_priors_random_dict(
        ::SampleSystemStaticCase; scenario = NTuple{0, Symbol}())
    #d_exp = Distributions.AffineDistribution(1, 1, Exponential(0.1))
    # prior in σ rather than σstar
    d_exp_lognorm = Exponential(log(1.05))
    d_exp_norm = Exponential(0.2)
    dd = Dict{Symbol, Distribution}([:a, :b1, :b2, :b3] .=> d_exp_norm)
    dd
end

function get_case_obs_uncertainty_dist_type(::SampleSystemStaticCase, stream;
        scenario = NTuple{0, Symbol}())
    dtypes = Dict{Symbol, Type}(:y => MvNormal)
    dtypes[stream]
end

function simulate_case_indivdata(inv_case::SampleSystemStaticCase; scenario)
    unc_par = Dict(:y => PDiagMat(fill(0.2, 12)))
    (; indivdata, p_indiv, d_noise) = simulate_indivdata(; 
        inv_case, scenario, unc_par, rng=StableRNG(0815))
end

function get_case_indivdata(
        inv_case::SampleSystemStaticCase, indiv_id; scenario = NTuple{0, Symbol}())
    (; indivdata, p_indiv) = simulate_case_indivdata(inv_case; scenario)
    indivdata[indiv_id]
end

function get_case_u0p(inv_case::SampleSystemStaticCase; scenario)
    df = DataFrame(indiv_id = Symbol[])
    if (:all_ranadd ∈ scenario) | (:all_ranmul ∈ scenario) 
        # set b3 to 0
        indiv_ids = get_case_indiv_ids(inv_case; scenario)
        ni = length(indiv_ids)
        df = DataFrame(;
            indiv_id = indiv_ids,
            u0 = fill(CA.ComponentVector(), ni),
            p = fill(CA.ComponentVector(b3 = 0.0), ni)
        )
    end
    if (:single_ranmul ∈ scenario) | (:single_ranadd ∈ scenario)
        # set b2 and b3 to 0
        indiv_ids = get_case_indiv_ids(inv_case; scenario)
        ni = length(indiv_ids)
        df = DataFrame(;
            indiv_id = indiv_ids,
            u0 = fill(CA.ComponentVector(), ni),
            p = fill(CA.ComponentVector(b2 = 0.0, b3 = 0.0), ni)
        )
    end
    df
end
