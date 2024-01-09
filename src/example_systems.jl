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

function samplesystem1(; name, τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])
    @parameters t
    D = Differential(t)
    @variables x1(..) x2(..) dec2(..) #dx(t)[1:2]  # observed dx now can be accessed
    ps = @parameters τ=τ i=i p[1:3]=p
    sts = [x1(t), x2(t), dec2(t)]
    eq = [
        D(x1(t)) ~ i - p[1] * x1(t) + (p[2] - x1(t)^2) / τ,
        D(x2(t)) ~ i - dec2(t),
        dec2(t) ~ p[3] * x2(t), # observable
    ]
    #ODESystem(eq, t; name)
    #ODESystem(eq, t, sts, [τ, p[1], p[2], i]; name)
    sys = ODESystem(eq, t, sts, vcat(ps...); name)
    #sys = ODESystem(eq, t, sts, ps; name)
    return sys
end

gen_site_data1 = () -> begin
    @named m1 = CP.samplesystem1()
    @named system = embed_system(m1)
    _dict_nums = get_system_symbol_dict(system)
    t = [0.2, 0.4, 1.0, 2.0]
    p_sites = ComponentVector(A = (m1₊x1 = 1.0, m1₊i = 4),
        B = (m1₊x1 = 1.2, m1₊i = 0.2))
    site = first(keys(p_sites))
    d_noise = fit(LogNormal, 1, Σstar(1.1))
    rng = StableRNG(123)
    obs_tuple = map(keys(p_sites)) do site
        st = Dict(m1.x1 .=> p_sites[site].m1₊x1, m1.x2 .=> 1.0)
        p_new = Dict(m1.i .=> p_sites[site].m1₊i)
        prob = ODEProblem(system, st, (0.0, 2.0), p_new)
        sol = solve(prob, Tsit5(), saveat = t)
        #sol[[m1.x[1], m1.dec2]]
        #sol[_dict_nums[:m1₊dec2]]
        streams = (:m1₊x1, :m1₊dec2)
        #stream = last(streams)
        tmp = map(streams) do stream
            obs_true = sol[_dict_nums[stream]]
            obs = obs_true .+ rand(rng, d_noise, length(obs_true))
            (; t, obs, obs_true)
        end
        (; zip(streams, tmp)...)
    end
    res = (; zip(keys(p_sites), obs_tuple)...)
    #clipboard(res) # not on slurm
    res  # copy from terminal and paste into get_sitedata
end

function get_sitedata(::Val{:CrossInverts_samplesystem1}, site; scenario)
    data = (A = (m1₊x1 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [2.4778914737556788, 2.8814759312054585,
                    3.1123382362789704, 3.319855891619185],
                obs_true = [
                    1.4943848813734615, 1.820680609257418,
                    2.229232180845057, 2.3324192686165475]),
            m1₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    2.9047245048321093, 3.487504088523498,
                    4.197993507626974, 4.729373001342233],
                obs_true = [
                    1.9181607097808226, 2.3947945842556075,
                    3.2641654938821785, 3.7994598198758065])),
        B = (m1₊x1 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    1.9072519083605397, 1.7328631403357329,
                    1.5368649796139198, 1.3767309450998682],
                obs_true = [
                    0.999447003144777, 0.8587653506110787,
                    0.6318569175937196, 0.5132152408929866]),
            m1₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    1.9267541330053009, 1.944429448795812,
                    1.709438971641723, 1.4449765000568517],
                obs_true = [
                    1.0481566696163267, 0.8539723454282818,
                    0.49978412872575745, 0.28170241145399255])))
    data[site]
end

function get_priors_dict(::Val{:CrossInverts_samplesystem1}, site; scenario)
    #using DataFrames, Tables, DistributionFits
    paramsModeUpperRows = [
        # τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])
        (:m1₊x1, LogNormal, 1.0, 2.0),
        (:m1₊x2, LogNormal, 1.0, 2.0),
        (:m1₊i, LogNormal, 1.0, 6.0),
        (:m1₊τ, LogNormal, 1.0, 5.0),
    ]
    df_scalars = df_from_paramsModeUpperRows(paramsModeUpperRows)
    dd = Dict{Symbol, Distribution}(df_scalars.par .=> df_scalars.dist)
    dist_p0 = fit(LogNormal, @qp_m(1.0), @qp_uu(3.0))
    dd[:m1₊p] = product_distribution(fill(dist_p0, 3))
    dd
end

function get_priors_random_dict(::Val{:CrossInverts_samplesystem1}; scenario)
    d_exp = Distributions.AffineDistribution(1, 1, Exponential(0.1))
    dd = Dict{Symbol, Distribution}([:m1₊x1,:m1₊x2,:m1₊i,:m1₊τ] .=> d_exp)
    dd[:sv₊p] = product_distribution(d_exp, d_exp, d_exp)
    dd
end

#--------------------------- samplesystem_vec -------------------------
function get_priors_dict(::Val{:CrossInverts_samplesystem_vec}, site; scenario)
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
    dd[:sv₊p] = product_distribution(fill(dist_p0, 3))
    dd[:sv₊x] = product_distribution(dd[:sv₊x_1], dd[:sv₊x_2])
    dd
end

function get_priors_random_dict(::Val{:CrossInverts_samplesystem_vec}; scenario)
    d_exp = Distributions.AffineDistribution(1, 1, Exponential(0.1))
    dd = Dict{Symbol, Distribution}([:sv₊τ, :sv₊i] .=> d_exp)
    dd[:sv₊x] = product_distribution(d_exp,d_exp)
    dd
end

function get_site_parameters(::Val{:CrossInverts_samplesystem1})
    @named sv = samplesystem_vec()
    @named system = embed_system(sv)
    scenario = :CrossInverts_samplesystem_vec
    #_dict_nums = get_system_symbol_dict(system)
    # setup a problem, numbers do not matter, because set below from prior mean
    t = [0.2, 0.4, 1.0, 2.0]
    p_siteA = ComponentVector(sv₊x = [1.1, 2.1], sv₊i = 4)
    st = Dict(Symbolics.scalarize(sv.x .=> p_siteA.sv₊x))
    p_new = Dict(sv.i .=> p_siteA.sv₊i)
    problem = ODEProblem(system, st, (0.0, 2.0), p_new)

    priors_dict = get_priors_dict(Val(scenario), :A; scenario=nothing)
    _m = Dict(k => mean(v) for (k,v) in priors_dict)
    fixed = ComponentVector(sv₊p = _m[:sv₊p])
    random = ComponentVector(sv₊x = _m[:sv₊x], sv₊τ = _m[:sv₊τ])
    indiv = ComponentVector(sv₊i = _m[:sv₊i])
    popt = vcat_statesfirst(fixed, random, indiv; system)
    psets = setup_psets_fixed_random_indiv(system, popt, keys(fixed), keys(random))
    pset = ODEProblemParSetter(system, popt)
    problem = remake(problem, popt, pset)

    p_A = ComponentVector(u0 = label_state(pset, problem.u0),
        p = label_par(pset, problem.p))
    # multiply random effects for sites B and C
    priors_random = dict_to_cv(keys(random), get_priors_random_dict(
        Val(scenario); scenario=nothing))
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


gen_site_data_vec = () -> begin
    p_sites = get_site_parameters(Val(:CrossInverts_samplesystem1))
    #using DistributionFits, StableRNGs, Statistics
    # other usings from test_util_mixed
    @named sv = CP.samplesystem_vec()
    @named system = embed_system(sv)
    scenario = :CrossInverts_samplesystem_vec
    _dict_nums = get_system_symbol_dict(system)
    # setup a problem, numbers do not matter, because set below from prior mean
    t = [0.2, 0.4, 1.0, 2.0]
    p_siteA = CA.ComponentVector(sv₊x = [1.1, 2.1], sv₊i = 4)
    st = Dict(Symbolics.scalarize(sv.x .=> p_siteA.sv₊x))
    p_new = Dict(sv.i .=> p_siteA.sv₊i)
    problem = ODEProblem(system, st, (0.0, 2.0), p_new)

    #site = first(keys(p_sites))
    rng = StableRNG(123)
    d_noise = Dict(:sv₊dec2 => fit(LogNormal, 1, Σstar(1.1)),
        :sv₊x => product_distribution([fit(LogNormal, 1, Σstar(1.1)),
            fit(LogNormal, 1, Σstar(1.1))]))
    obs_tuple = map(keys(p_sites)) do site
        #st = Dict(Symbolics.scalarize(sv.x .=> p_sites[site].u0.sv₊x))
        #p_new = Dict(sv.i .=> p_sites[site].sv₊i)
        #prob = ODEProblem(system, st, (0.0, 2.0), p_new)
        probo = remake(problem,
            u0 = CA.getdata(p_sites[site].u0), p = CA.getdata(p_sites[site].p))
        sol = solve(probo, Tsit5(), saveat = t)
        #sol[[sv.x[1], sv.dec2]]
        #sol[_dict_nums[:sv₊dec2]]
        streams = (:sv₊x, :sv₊dec2)
        #stream = last(streams) #stream = first(streams)
        tmp = map(streams) do stream
            obs_true = sol[Symbolics.scalarize(_dict_nums[stream])]
            noise = rand(rng, d_noise[stream], length(obs_true))
            obs = length(size(noise)) == 1 ?
                  obs = obs_true .+ noise :
                  obs = obs_true .+ eachcol(noise)
            (; t, obs, obs_true)
        end
        (; zip(streams, tmp)...)
    end
    res = (; zip(keys(p_sites), obs_tuple)...)
    #clipboard(res) # not on slurm
    res  # copy from terminal and paste into get_sitedata
end

function get_sitedata(::Val{:CrossInverts_samplesystem_vec}, site; scenario)
    data = (A = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
    obs = [
        [1.9521927604522915, 2.047517187862125],
        [1.959751009453174, 2.1821024478735236],
        [1.994686092440237, 2.255455473386112],
        [2.2082422198908183, 2.2818243329314463],
    ],
    obs_true = [
        [1.1107462105660761, 1.1132307256263168],
        [1.1374030624789362, 1.1517682443665676],
        [1.1662811048774226, 1.2254956639115882],
        [1.1723596106288048, 1.2754477749552304],
    ]),
sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
    obs = [
        2.3735296942998776, 2.382273373816772,
        2.341284913075192, 2.5324964004269113,
    ],
    obs_true = [
        1.3025989791594734, 1.3476919966398768,
        1.4339609606781936, 1.4924102716381489,
    ])),
B = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
    obs = [
        [2.1179381383169464, 2.1210382245945834],
        [2.0822012511935855, 2.0674226074688833],
        [2.1502642025499505, 2.306549239404262],
        [2.1769831651053204, 2.2252801829256583],
    ],
    obs_true = [
        [1.151651990792978, 1.1132958972995004],
        [1.1625064462552526, 1.151819832008928],
        [1.1747525567182713, 1.2255212968919122],
        [1.1775878195616543, 1.2754555091009407],
    ]),
sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
    obs = [
        2.245788014365762, 2.277509740281726,
        2.4334715751574127, 2.5137082813474017,
    ],
    obs_true = [
        1.3026752369854613, 1.3477523596973535,
        1.4339909540059328, 1.4924193214155894,
    ])),
C = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
    obs = [
        [2.0201527203706373, 2.073290144592896],
        [2.3016809589613776, 2.2358258450652864],
        [2.075680650697116, 2.104109424435813],
        [2.1132474277961473, 2.2382429293126203],
    ],
    obs_true = [
        [1.0937218361731353, 1.1239461365370844],
        [1.1276412608646247, 1.1602478223825532],
        [1.1645958587429655, 1.229697745239713],
        [1.1724268272073743, 1.2767519138766517],
    ]),
sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
    obs = [
        2.2606131416433883, 2.3903847629996546,
        2.340386125399106, 2.3963968314981225,
    ],
    obs_true = [
        1.3151371556509526, 1.3576140095820832,
        1.438877845132026, 1.493936253618901,
    ])))
    data[site]
end

function map_keys(FUN, cv::ComponentVector; rewrap::Val{is_rewrap}=Val(true)) where is_rewrap
    tup = map(keys(cv)) do k
        FUN(cv[k])
    end
    is_rewrap ? ComponentVector(;zip(keys(cv),tup)...) : tup
end




