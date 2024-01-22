struct SampleSystem1Case <: AbstractCrossInversionCase end

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
    p_indiv = ComponentVector(A = (m1₊x1 = 1.0, m1₊i = 4),
        B = (m1₊x1 = 1.2, m1₊i = 0.2))
    indiv_id = first(keys(p_indiv))
    d_noise = fit(LogNormal, 1, Σstar(1.1))
    rng = StableRNG(123)
    obs_tuple = map(keys(p_indiv)) do indiv_id
        st = Dict(m1.x1 .=> p_indiv[indiv_id].m1₊x1, m1.x2 .=> 1.0)
        p_new = Dict(m1.i .=> p_indiv[indiv_id].m1₊i)
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
    res = (; zip(keys(p_indiv), obs_tuple)...)
    #clipboard(res) # not on slurm
    res  # copy from terminal and paste into get_indivdata
end

function get_indivdata(::SampleSystem1Case, indiv_id; scenario = NTuple{0,Symbol}())
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
    data[indiv_id]
end

function get_priors_dict(::SampleSystem1Case, indiv_id; scenario = NTuple{0,Symbol}())
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

function get_priors_random_dict(::SampleSystem1Case; scenario = NTuple{0,Symbol}())
    #d_exp = Distributions.AffineDistribution(1, 1, Exponential(0.1))
    # prior in σ rather than σstar
    d_exp = Exponential(log(1.05))
    dd = Dict{Symbol, Distribution}([:m1₊x1,:m1₊x2,:m1₊i,:m1₊τ] .=> d_exp)
    dd[:m1₊p] = product_distribution(d_exp, d_exp, d_exp)
    dd
end
