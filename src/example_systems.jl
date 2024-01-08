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
    @named sys = embed_system(m1)
    _dict_nums = get_system_symbol_dict(sys)
    t = [0.2, 0.4, 1.0, 2.0]
    p_sites = ComponentVector(A = (m1₊x1 = 1.0, m1₊i = 4),
        B = (m1₊x1 = 1.2, m1₊i = 0.2))
    site = first(keys(p_sites))
    d_noise = fit(LogNormal, 1, Σstar(1.1))
    rng = StableRNG(123)
    obs_tuple = map(keys(p_sites)) do site
        st = Dict(m1.x1 .=> p_sites[site].m1₊x1, m1.x2 .=> 1.0)
        p_new = Dict(m1.i .=> p_sites[site].m1₊i)
        prob = ODEProblem(sys, st, (0.0, 2.0), p_new)
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

function get_sitedata(::Val{:CrossInverts_samplesystem1}, site, scenario)
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

function get_priors_df(::Val{:CrossInverts_samplesystem1}, site, scenario)
    #using DataFrames, Tables, DistributionFits
    paramsModeUpperRows = [
        # τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])
        (:m1₊x1, LogNormal, 1.0, 2.0),
        (:m1₊x2, LogNormal, 1.0, 2.0),
        (:m1₊i, LogNormal, 1.0, 6.0),
        (:m1₊τ, LogNormal, 1.0, 5.0),
    ]
    df_scalars = df_from_paramsModeUpperRows(paramsModeUpperRows)
    dist_p0 = fit(LogNormal, @qp_m(1.0), @qp_uu(3.0))
    dist_p = product_distribution(fill(dist_p0, 3))
    df_p = DataFrame(par = :m1₊p,
        dType = Product,
        med = missing,
        upper = missing,
        dist = dist_p)
    vcat(df_scalars, df_p)
end

gen_site_data_vec = () -> begin
    #using DistributionFits, StableRNGs
    @named sv = CP.samplesystem_vec()
    @named sys = embed_system(sv)
    _dict_nums = get_system_symbol_dict(sys)
    t = [0.2, 0.4, 1.0, 2.0]
    p_sites = CA.ComponentVector(A = (sv₊x = [1.1, 2.1], sv₊i = 4),
        B = (sv₊x = [1.2, 2.2], sv₊i = 0.2))
    site = first(keys(p_sites))
    d_noise = Dict(:sv₊dec2 => fit(LogNormal, 1, Σstar(1.1)),
        :sv₊x => product_distribution([fit(LogNormal, 1, Σstar(1.1)),
            fit(LogNormal, 1, Σstar(1.1))]))
    rng = StableRNG(123)
    obs_tuple = map(keys(p_sites)) do site
        st = Dict(Symbolics.scalarize(sv.x .=> p_sites[site].sv₊x))
        p_new = Dict(sv.i .=> p_sites[site].sv₊i)
        prob = ODEProblem(sys, st, (0.0, 2.0), p_new)
        sol = solve(prob, Tsit5(), saveat = t)
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

function get_sitedata(::Val{:CrossInverts_samplesystem_vec}, site, scenario)
    data = (A = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [2.5453299001305467, 3.384460308262885],
                    [2.746948743128864, 3.4835588561848265],
                    [3.2254856047045917, 3.903390612822624],
                    [3.266940736476312, 3.93427603779292],
                ],
                obs_true = [
                    [1.5618233077483297, 2.323664986314844],
                    [1.86384268769495, 2.496122233182189],
                    [2.238921809653305, 2.810681108554734],
                    [2.333112722731517, 3.0043628563264932],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    3.92856938742506, 4.1190566928615,
                    4.558893503141354, 4.769187417431323,
                ],
                obs_true = [
                    3.020764482209297, 3.2449589031368458,
                    3.653885441121154, 3.905671713224441,
                ])),
        B = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [1.8780444827225167, 2.8219932941025663],
                    [2.068420236029813, 2.5336002971521934],
                    [1.6869699748371052, 1.7602460722708764],
                    [1.4780942191074935, 1.2543547914839175],
                ],
                obs_true = [
                    [0.9994470193335424, 1.7315361907350357],
                    [0.8587653931138475, 1.3703262085493344],
                    [0.6318570996015761, 0.7114868532943917],
                    [0.5132149571227039, 0.305823617777594],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    3.3506468414438086, 2.685921077040386,
                    1.869960763222208, 1.4577056889312732,
                ],
                obs_true = [
                    2.2509970479555466, 1.7814240711141347,
                    0.9249329092827092, 0.39757070311087217,
                ])))
    data[site]
end

function get_priors_df(::Val{:CrossInverts_samplesystem_vec}, site, scenario)
    #using DataFrames, Tables, DistributionFits, Chain
    paramsModeUpperRows = [
        # τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])
        (:sv₊i, LogNormal, 1.0, 6.0),
        (:sv₊τ, LogNormal, 1.0, 5.0),
    ]
    df_scalars = df_from_paramsModeUpperRows(paramsModeUpperRows)
    # TODO par, index, ....
    paramsModeUpperRows_multivariate = [
        (:sv₊x_1, LogNormal, 1.0, 2.0),
        (:sv₊x_2, LogNormal, 1.0, 2.0),
    ]
    df_mv = df_from_paramsModeUpperRows(paramsModeUpperRows_multivariate)
    dist_p0 = fit(LogNormal, @qp_m(1.0), @qp_uu(3.0))
    dist_p = product_distribution(fill(dist_p0, 3))
    df_p = DataFrame(par = :sv₊p,
        dType = Product,
        med = missing,
        upper = missing,
        dist = dist_p)
    dist_x = @chain df_mv begin
        filter(:par => in((:sv₊x_1, :sv₊x_2)), _)
        _.dist
        product_distribution()
    end
    df_x = DataFrame(par = :sv₊x,
        dType = Product,
        med = missing,
        upper = missing,
        dist = dist_x)
    vcat(df_scalars, df_p, df_x)
end

