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

function get_priors_dict(::SampleSystemVecCase, site; scenario = NTuple{0,Symbol}())
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

function get_site_parameters(inv_case::SampleSystemVecCase; scenario = NTuple{0,Symbol}())
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

    p_A = ComponentVector(u0 = label_state(pset, problem.u0),
        p = label_par(pset, problem.p))
    # multiply random effects for sites B and C
    priors_random = dict_to_cv(keys(random), get_priors_random_dict(
        inv_case; scenario))
    rng = StableRNG(234)
    _get_u0p_ranef = () -> begin
        probo = sample_and_add_ranef(problem, priors_random, rng; psets)
        ComponentVector(u0 = label_state(pset, probo.u0), p = label_par(pset, probo.p))
    end
    _get_u0p_ranef()
    p_sites = ComponentVector(A=p_A, B=_get_u0p_ranef(), C=_get_u0p_ranef())
    # modify fixed parameters of third site
    p_sites.C.p.sv₊p = p_sites.C.p.sv₊p .* 1.2
    p_sites
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
    inv_case = SampleSystemVecCase()
    scenario = NTuple{0,Symbol}()
    p_sites = CP.get_site_parameters(inv_case)
    #using DistributionFits, StableRNGs, Statistics
    # other usings from test_util_mixed
    @named sv = CP.samplesystem_vec()
    @named system = embed_system(sv)
    _dict_nums = get_system_symbol_dict(system)
    # setup a problem, numbers do not matter, because set below from prior mean
    t = [0.2, 0.4, 1.0, 2.0]
    p_siteA = p_sites.A 
    st = p_siteA.u0
    #st = Dict(Symbolics.scalarize(sv.x .=> p_siteA.sv₊x))
    p_new = Dict(sv.i .=> p_siteA.p.sv₊i)
    problem = ODEProblem(system, st, (0.0, 2.0), p_new)
    #site = first(keys(p_sites))
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
    # site = first(keys(p_sites))
    obs_tuple = map(keys(p_sites)) do site
        #st = Dict(Symbolics.scalarize(sv.x .=> p_sites[site].u0.sv₊x))
        #p_new = Dict(sv.i .=> p_sites[site].sv₊i)
        #prob = ODEProblem(system, st, (0.0, 2.0), p_new)
        probo = remake(problem,
            u0 = CA.getdata(p_sites[site].u0), p = CA.getdata(p_sites[site].p))
        sol = solve(probo, Tsit5(), saveat = t)
        #sol[[sv.x[1], sv.dec2]]
        #sol[_dict_nums[:sv₊dec2]]
        #stream = last(streams) #stream = first(streams)
        tmp = map(streams) do stream
            obs_true = sol[Symbolics.scalarize(_dict_nums[stream])]
            n_obs = length(obs_true)
            obs_unc = fill(unc_par[stream], n_obs)  # may be diffrent for each obs
            noise = rand(rng, d_noise[stream], n_obs)
            obs = length(size(noise)) == 1 ?
                  obs = obs_true .+ noise :
                  obs = obs_true .+ eachcol(noise)
            (; t, obs, obs_unc, obs_true)
        end
        (; zip(streams, tmp)...)
    end
    res = (; zip(keys(p_sites), obs_tuple)...)
    #clipboard(res) # not on slurm
    res  # copy from terminal and paste into get_sitedata
end

function get_sitedata(::SampleSystemVecCase, site; scenario = NTuple{0,Symbol}())
    data = (A = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [2.0275948679676405, 2.2846639923962027],
                    [1.7842789336214449, 2.0805372364458754],
                    [2.092393487998817, 2.51497820774601],
                    [1.9475052055463449, 2.0401167718843545],
                ],
                obs_unc = [[0.09531017980432493 0.0;
                        0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [
                    [1.1107462105660761, 1.1132307256263168],
                    [1.1374030624789362, 1.1517682443665676],
                    [1.1662811048774226, 1.2254956639115882],
                    [1.1723596106288048, 1.2754477749552304],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    1.4910499737220133,
                    1.4694637527792267,
                    1.6158188934360207,
                    1.5982127394156822,
                ],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [
                    1.3025989791594734,
                    1.3476919966398768,
                    1.4339609606781936,
                    1.4924102716381489,
                ])),
        B = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [1.7878914495255218, 2.3941886176710527],
                    [2.9549503194333533, 2.7310549238893786],
                    [2.3259814660317266, 2.3544460233904667],
                    [2.0393725877635633, 2.0908352990694827],
                ],
                obs_unc = [[0.09531017980432493 0.0;
                        0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [
                    [1.151651990792978, 1.1132958972995004],
                    [1.1625064462552526, 1.151819832008928],
                    [1.1747525567182713, 1.2255212968919122],
                    [1.1775878195616543, 1.2754555091009407],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    3.0251216726883907,
                    1.5284285861011693,
                    1.7336432112674052,
                    2.6214812237491136,
                ],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [
                    1.3026752369854613,
                    1.3477523596973535,
                    1.4339909540059328,
                    1.4924193214155894,
                ])),
        C = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [1.8350702176355616, 1.8894285743818107],
                    [1.9539128916619513, 1.852396055293947],
                    [2.1408743696478636, 1.8504990396521535],
                    [1.945593593771847, 1.713264714840722],
                ],
                obs_unc = [[0.09531017980432493 0.0;
                        0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [
                    [1.0810227295857942, 1.07893746485692],
                    [1.1064444715184476, 1.0795904809250938],
                    [1.1323810742565483, 1.0807372211957795],
                    [1.137221558052231, 1.0813915618203442],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    1.9344200864095762,
                    1.898790749962489,
                    5.962841895842549,
                    3.4042330987378318,
                ],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [
                    1.514966636777468,
                    1.5158835551242544,
                    1.517493725600046,
                    1.5184125037939566,
                ])))
    data[site]
end
