struct SampleSystemVecCase <: AbstractCrossInversionCase end

function samplesystem_vec(; name, τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])
    n_comp = 2
    @parameters t
    D = Differential(t)
    @variables x(..)[1:n_comp] dec2(..) #dx(t)[1:2]  # observed dx now can be accessed
    #sts = @variables x[1:n_comp](t) 
    #ps = @parameters τ=τ p[1:n_comp]=p i=i       # parameters
    ps = @parameters τ=τ i=i p[1:3]=p
    sts = vcat([x(t)[i] for i in 1:n_comp], dec2(t))
    eq = [
        D(x(t)[1]) ~ i - p[1] * x(t)[1] + (p[2] - x(t)[1]^2) / τ,
        D(x(t)[2]) ~ i - dec2(t),
        dec2(t) ~ p[3] * x(t)[2], # observable
    ]
    #ODESystem(eq, t; name)
    #ODESystem(eq, t, sts, [τ, p[1], p[2], i]; name)
    sys = ODESystem(eq, t, sts, vcat(ps...); name)
    #sys = ODESystem(eq, t, sts, ps; name)
    return sys
end

function product_MvLogNormal(comp...)
    μ = collect(getproperty.(comp, :μ))
    σ = collect(getproperty.(comp, :σ))
    Σ = PDiagMat(exp.(σ))
    MvLogNormal(μ, Σ)
end

function get_priors_dict(::SampleSystemVecCase, indiv_id; scenario = NTuple{0,Symbol}())
    #using DataFrames, Tables, DistributionFits, Chain
    paramsModeUpperRows = [
        # τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])
        (:sv₊i, LogNormal, 1.0, 6.0),
        (:sv₊τ, LogNormal, 1.0, 5.0),
        (:sv₊x_1, LogNormal, 1.0, 2.0),
        (:sv₊x_2, LogNormal, 1.0, 2.0),
    ]
    df_scalars = df_from_paramsModeUpperRows(paramsModeUpperRows)
    dd = Dict{Symbol, Distribution}(df_scalars.par .=> df_scalars.dist)
    dist_p0 = fit(LogNormal, @qp_m(1.0), @qp_uu(3.0))
    # dd[:sv₊p] = product_distribution(fill(dist_p0, 3))
    # dd[:sv₊x] = product_distribution(dd[:sv₊x_1], dd[:sv₊x_2])
    dd[:sv₊p] = product_MvLogNormal(fill(dist_p0, 3)...)
    dd[:sv₊x] = product_MvLogNormal(dd[:sv₊x_1], dd[:sv₊x_2])
    dd
end

function get_priors_random_dict(::SampleSystemVecCase; scenario = NTuple{0,Symbol}())
    #d_exp = Distributions.AffineDistribution(1, 1, Exponential(0.1))
    # prior in σ rather than σstar
    d_exp = Exponential(log(1.05))
    dd = Dict{Symbol, Distribution}([:sv₊τ, :sv₊i] .=> d_exp)
    # https://github.com/TuringLang/Bijectors.jl/issues/300
    # dd[:sv₊x] = product_distribution(d_exp,d_exp)
    dd[:sv₊x] = Distributions.Product(fill(d_exp,2))
    # d_lognorm = fit(LogNormal, moments(d_exp))
    # dd[:sv₊x] = product_MvLogNormal(d_lognorm,d_lognorm)
    dd
end

function get_indiv_parameters(inv_case::SampleSystemVecCase; scenario = NTuple{0,Symbol}())
    @named sv = samplesystem_vec()
    @named system = embed_system(sv)
    #_dict_nums = get_system_symbol_dict(system)
    # setup a problem, numbers do not matter, because set below from prior mean
    t = [0.2, 0.4, 1.0, 2.0]
    p_siteA = ComponentVector(sv₊x = [1.1, 2.1], sv₊i = 4)
    st = Dict(Symbolics.scalarize(sv.x .=> p_siteA.sv₊x))
    p_new = Dict(sv.i .=> p_siteA.sv₊i)
    problem = ODEProblem(system, st, (0.0, 2.0), p_new)

    priors_dict = get_priors_dict(inv_case, :A; scenario)
    _m = Dict(k => mean(v) for (k,v) in priors_dict)
    fixed = ComponentVector(sv₊p = _m[:sv₊p])
    random = ComponentVector(sv₊x = _m[:sv₊x], sv₊τ = _m[:sv₊τ])
    indiv = ComponentVector(sv₊i = _m[:sv₊i])
    popt = vcat_statesfirst(fixed, random, indiv; system)
    psets = setup_psets_fixed_random_indiv(keys(fixed), keys(random); system, popt)
    pset = ODEProblemParSetter(system, popt)
    problem = remake(problem, popt, pset)

    p_A = (label_state(pset, problem.u0), label_par(pset, problem.p)) # Tuple (u0, p)
    # multiply random effects for sites B and C
    priors_random = dict_to_cv(keys(random), get_priors_random_dict(
        inv_case; scenario))
    rng = StableRNG(234)
    _get_u0p_ranef = () -> begin
        probo = sample_and_add_ranef(problem, priors_random, rng; psets)
        (label_state(pset, probo.u0), label_par(pset, probo.p))
    end
    #_get_u0p_ranef()
    p_indiv = rename(DataFrame([
        (:A, p_A...),
        (:B, _get_u0p_ranef()...), 
        (:C, _get_u0p_ranef()...)
        ]),["indiv_id","u0","p"])
    # ComponentVector(A=p_A, B=_get_u0p_ranef(), C=_get_u0p_ranef())
    if :modify_fixed ∈ scenario
        # modify fixed parameters of third indiv_id
        p_indiv.p[3].sv₊p = p_indiv.p[3].sv₊p .* 1.05
    end
    p_indiv
end

function get_obs_uncertainty_dist_type(::SampleSystemVecCase, stream; 
    scenario = NTuple{0,Symbol}())
    dtypes = Dict{Symbol, Type}(
        :sv₊dec2 => LogNormal,
        :sv₊x => MvLogNormal,
    )
    dtypes[stream]
end


gen_site_data_vec = () -> begin
    # using in test_util_mixed
    inv_case = SampleSystemVecCase()
    scenario = NTuple{0,Symbol}()
    p_indiv = CP.get_indiv_parameters(inv_case)
    #using DistributionFits, StableRNGs, Statistics
    # other usings from test_util_mixed
    @named sv = CP.samplesystem_vec()
    @named system = embed_system(sv)
    _dict_nums = get_system_symbol_dict(system)
    # setup a problem, numbers do not matter, because set below from prior mean
    t = [0.2, 0.4, 1.0, 2.0]
    u0_A, p_A = (p_indiv.u0[1], p_indiv.p[1])
    p_new = Dict(sv.i .=> p_A.sv₊i)
    problem = ODEProblem(system, u0_A, (0.0, 2.0), p_new)
    #indiv_id = first(keys(p_indiv))
    streams = (:sv₊x, :sv₊dec2)
    dtypes = Dict(s => get_obs_uncertainty_dist_type(inv_case, s; scenario) for s in streams)
    unc_par = Dict(:sv₊dec2 => 1.1, :sv₊x => PDiagMat(log.([1.1,1.1])))
    d_noise = Dict(s => begin
        unc = unc_par[s]
        m = unc isa AbstractMatrix ? fill(1.0, size(unc,1)) : 1.0
        fit_mean_Σ(dtypes[s], m, unc) 
    end for s in streams)
    # d_noise[:sv₊x]
    rng = StableRNG(123)
    indiv_dict = Dict(p_indiv.indiv_id .=> zip(p_indiv.u0, p_indiv.p))
    # indiv_id = first(p_indiv.indiv_id)
    obs_tuple = map(p_indiv.indiv_id) do indiv_id
        #st = Dict(Symbolics.scalarize(sv.x .=> p_indiv[indiv_id].u0.sv₊x))
        #p_new = Dict(sv.i .=> p_indiv[indiv_id].sv₊i)
        #prob = ODEProblem(system, st, (0.0, 2.0), p_new)
        probo = remake(problem,
            u0 = CA.getdata(indiv_dict[indiv_id][1]), p = CA.getdata(indiv_dict[indiv_id][2]))
        sol = solve(probo, Tsit5(), saveat = t)
        #sol[[sv.x[1], sv.dec2]]
        #sol[_dict_nums[:sv₊dec2]]
        #stream = last(streams) #stream = first(streams)
        tmp = map(streams) do stream
            obs_true = sol[Symbolics.scalarize(_dict_nums[stream])]
            n_obs = length(obs_true)
            obs_unc = fill(unc_par[stream], n_obs)  # may be different for each obs
            noise = rand(rng, d_noise[stream], n_obs)
            obs = length(size(noise)) == 1 ?
                  obs = obs_true .+ noise :
                  obs = obs_true .+ eachcol(noise)
            (; t, obs, obs_unc, obs_true)
        end
        (; zip(streams, tmp)...)
    end
    res = (; zip(p_indiv.indiv_id, obs_tuple)...)
    #clipboard(res) # not on slurm
    res  # copy from terminal and paste into get_indivdata
end

function get_indivdata(::SampleSystemVecCase, indiv_id; scenario = NTuple{0, Symbol}())
    data = (A = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [2.3696993004601956, 2.673733320916141],
                    [1.8642844249865063, 2.0994355527637607],
                    [1.9744553950945931, 2.049494086682751],
                    [1.806115091024414, 1.4088107777562726],
                ],
                obs_unc = [[0.09531017980432493 0.0;
                        0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [
                    [1.4528506430586314, 1.502300054146255],
                    [1.2174085538439976, 1.1706665606844529],
                    [1.0483430119731987, 0.7600115428483291],
                    [1.0309694961068738, 0.6441417808271487],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    3.7951565919532038,
                    2.932295276687423,
                    2.0064853619502925,
                    1.6522510350996853,
                ],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [
                    3.606705597390664,
                    2.810523520548073,
                    1.8246274291924653,
                    1.546448567322152,
                ])),
        B = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [1.3716397862102392, 1.7915277956458109],
                    [2.685917136119328, 2.136372707813559],
                    [2.1722359214511475, 1.7436450983505765],
                    [1.8964297976426767, 1.4463477610356354],
                ],
                obs_unc = [[0.09531017980432493 0.0;
                        0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [
                    [0.7354003274776955, 0.5106350752742588],
                    [0.8934732629412274, 0.5571376159331084],
                    [1.0210070121376924, 0.6147203718520219],
                    [1.0346450294407676, 0.6309679710670936],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    2.9483735593375626,
                    1.5182461428508836,
                    1.775466226561185,
                    2.6438829307566296,
                ],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [
                    1.2259271236346332,
                    1.337569916447068,
                    1.4758139692997125,
                    1.5148210284231054,
                ])),
        C = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [2.408717132341871, 1.8247536086704808],
                    [2.147234967910222, 1.6415300804716022],
                    [2.054292483902587, 1.458270046707626],
                    [1.8269139195451993, 1.269531560860636],
                ],
                obs_unc = [[0.09531017980432493 0.0;
                        0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [
                    [1.6546696442921036, 1.0142624991455902],
                    [1.2997665477667184, 0.8687245061027491],
                    [1.045799188511272, 0.688508228251252],
                    [1.0185418838255835, 0.6376584078402582],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    2.85448380378512,
                    2.4685315214191292,
                    6.098311221019019,
                    3.4167039510572197,
                ],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [
                    2.4350303541530116,
                    2.085624326580895,
                    1.6529630507765165,
                    1.5308833561133446,
                ])))
    data[indiv_id]
end
