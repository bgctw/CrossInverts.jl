struct SampleSystemVecCase <: AbstractCrossInversionCase end

function samplesystem_vec(; name, τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3], b1 = 0.01)
    n_comp = 2
    @parameters t
    D = Differential(t)
    @variables x(..)[1:n_comp] dec2(..) b1obs(..) #dx(t)[1:2]  
    #sts = @variables x[1:n_comp](t) 
    #ps = @parameters τ=τ p[1:n_comp]=p i=i       # parameters
    ps = @parameters τ=τ i=i i2 p[1:3]=p b1=b1
    sts = vcat([x(t)[i] for i in 1:n_comp], dec2(t), b1obs(t))
    eq = [
        D(x(t)[1]) ~ i - p[1] * x(t)[1] + (p[2] - x(t)[1]^2) / τ,
        D(x(t)[2]) ~ i - dec2(t) + i2,
        dec2(t) ~ p[3] * x(t)[2],  # observable
        b1obs(t) ~ b1
    ]
    #ODESystem(eq, t; name)
    #ODESystem(eq, t, sts, [τ, p[1], p[2], i]; name)
    sys = ODESystem(eq, t, sts, vcat(ps...); name)
    #sys = ODESystem(eq, t, sts, ps; name)
    return sys
end

function get_case_inverted_system(::SampleSystemVecCase; scenario)
    @named sv = samplesystem_vec()
    @named system = embed_system(sv)
    u0_default = ComponentVector{Float64}()
    p_default = ComponentVector(sv₊i2 = 0.1)
    (; system, u0_default, p_default)
end

get_case_indiv_ids(::SampleSystemVecCase; scenario) = (:A, :B, :C)

function get_case_mixed_keys(::SampleSystemVecCase; scenario)
    (;
        fixed = (:sv₊p,),
        ranadd = (:sv₊b1,),
        ranmul = (:sv₊x, :sv₊τ),
        indiv = (:sv₊i,))
end

function get_case_priors_dict(
        ::SampleSystemVecCase, indiv_id; scenario = NTuple{0, Symbol}())
    #using DataFrames, Tables, DistributionFits, Chain
    paramsModeUpperRows = [
        # τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])
        (:sv₊i, LogNormal, 1.0, 6.0),
        (:sv₊τ, LogNormal, 1.0, 5.0),
        (:sv₊x_1, LogNormal, 1.0, 2.0),
        (:sv₊x_2, LogNormal, 1.0, 2.0),
        (:sv₊b1, LogNormal, 0.01, 0.3)
    ]
    if :test_indiv_priors ∈ scenario
        # test setting a different indiv prior for site B
        if indiv_id == :B
            paramsModeUpperRows[1] = (:sv₊i, LogNormal, 2.0, 6.0)
        end
    end
    df_scalars = df_from_paramsModeUpperRows(paramsModeUpperRows)
    dd = Dict{Symbol, Distribution}(df_scalars.par .=> df_scalars.dist)
    dist_p0 = fit(LogNormal, 1.0, @qp_uu(3.0), Val(:mode))
    # dd[:sv₊p] = product_distribution(fill(dist_p0, 3))
    # dd[:sv₊x] = product_distribution(dd[:sv₊x_1], dd[:sv₊x_2])
    dd[:sv₊p] = product_MvLogNormal(fill(dist_p0, 3)...)
    dd[:sv₊x] = product_MvLogNormal(dd[:sv₊x_1], dd[:sv₊x_2])
    return(dd)
end

function product_MvLogNormal(comp...)
    μ = collect(getproperty.(comp, :μ))
    σ = collect(getproperty.(comp, :σ))
    Σ = PDiagMat(exp.(σ))
    MvLogNormal(μ, Σ)
end

function get_case_priors_random_dict(::SampleSystemVecCase; scenario = NTuple{0, Symbol}())
    #d_exp = Distributions.AffineDistribution(1, 1, Exponential(0.1))
    # prior in σ rather than σstar
    d_exp = Exponential(log(1.05))
    dd = Dict{Symbol, Distribution}([:sv₊τ, :sv₊i, :sv₊b1] .=> d_exp)
    # https://github.com/TuringLang/Bijectors.jl/issues/300
    dd[:sv₊x] = product_distribution(d_exp, d_exp)
    #dd[:sv₊x] = Distributions.Product(fill(d_exp, 2))
    # d_lognorm = fit(LogNormal, moments(d_exp))
    # dd[:sv₊x] = product_MvLogNormal(d_lognorm,d_lognorm)
    return(dd)
end

function get_case_obs_uncertainty_dist_type(::SampleSystemVecCase, stream;
        scenario = NTuple{0, Symbol}())
    dtypes = Dict{Symbol, Type}(:sv₊dec2 => LogNormal,
        :sv₊x => MvLogNormal, :sv₊b1obs => Normal)
    dtypes[stream]
end

function get_case_indivdata(::SampleSystemVecCase, indiv_id; scenario = NTuple{0, Symbol}())
    # generated as in test_indivdata_vec.jl testset simulate_indivdata
    data = (
        A = (
            sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [[1.4912885324684522, 2.2409102326644423],
                    [1.0219115166133077, 2.4038379561438],
                    [0.980866526634816, 5.50540437766273],
                    [1.874575732881241, 13.961758721461388]],
                obs_unc = [[0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [[1.4787292481600516, 2.91117353307833],
                    [1.264159905907183, 3.804304621352185],
                    [1.107798229630297, 6.378980629352097],
                    [1.0914837982410082, 10.341246155610552]]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [1.8689071090748612, 0.41654164615739786,
                    0.8093691207865141, 3.233641140651625],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [0.291117353307833, 0.38043046213521853,
                    0.6378980629352098, 1.0341246155610553]),
            sv₊b1obs = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [-0.7795167652502649, -0.05741424425740792,
                    -0.5888288516959914, -0.2964700876107901],
                obs_unc = [0.5, 0.5, 0.5, 0.5],
                obs_true = [0.06315476261390056, 0.06315476261390056,
                    0.06315476261390056, 0.06315476261390056])),
        B = (
            sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [[1.823477539192328, 3.3573213689144774],
                    [1.4075389013592818, 5.201440985981608],
                    [0.41154505861414775, 9.90177720253818],
                    [1.2232618612254267, 8.948825104108398]],
                obs_unc = [[0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [[1.4760666098369952, 2.9111735330782675],
                    [1.2623973384218, 3.804304621352342],
                    [1.10849921946697, 6.378980629361265],
                    [1.0928205247778953, 10.341246155611023]]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [0.7913328673739444, 0.7832349437438053,
                    0.4143988518284921, 0.35319475381747517],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [0.2911173533078268, 0.38043046213523424,
                    0.6378980629361265, 1.0341246155611024]),
            sv₊b1obs = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [1.0919735356273452, 0.36937993886123344,
                    0.39056862830907546, -0.2546093775136992],
                obs_unc = [0.5, 0.5, 0.5, 0.5],
                obs_true = [0.06404680515614612, 0.06404680515614612,
                    0.06404680515614612, 0.06404680515614612])),
        C = (
            sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [[1.2003442185641153, 2.6004654556021],
                    [1.104253251555439, 3.634309670783337],
                    [1.4151319519309853, 7.253516704188803],
                    [1.1755348603995945, 9.761539934187418]],
                obs_unc = [[0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [[1.5055650158158165, 3.031535169955061],
                    [1.3018890303900401, 4.04264457501199],
                    [1.156812220691872, 6.957422887887796],
                    [1.1427128334290835, 11.443084613853932]]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [0.20670597497397553, 0.6185539915639261,
                    1.333568599014164, 1.833092592901468],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [0.30315351699550613, 0.40426445750119905,
                    0.6957422887887796, 1.1443084613853933]),
            sv₊b1obs = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [0.11585880669585569, 0.24786815768591602,
                    0.5437709057237933, 0.6715237531313522],
                obs_unc = [0.5, 0.5, 0.5, 0.5],
                obs_true = [0.05001340490218502, 0.05001340490218502,
                    0.05001340490218502, 0.05001340490218502])))
    data[indiv_id]
end

function get_case_problemupdater(
        ::SampleSystemVecCase; system, scenario = NTuple{0, Symbol}())
    mapping = (:sv₊i => :sv₊i2,)
    pset = ODEProblemParSetter(system, Symbol[]) # parsetter to get state symbols
    get_ode_problemupdater(KeysProblemParGetter(mapping, keys(axis_state(pset))), system)
end


function get_case_u0p(::SampleSystemVecCase; scenario)
    # creating the csv string:
    # io = IOBuffer()
    # write_csv_cv(io, indiv_info[:, 1:3])
    # s = String(take!(io))
    # print(s)
    csv = """
# u0=(Axis(sv₊x = 1:2,),)
# p=(Axis(sv₊τ = 1, sv₊i = 2, sv₊i2 = 3, sv₊p = 4:6, sv₊b1 = 7),)
indiv_id,u0_1,u0_2,p_1,p_2,p_3,p_4,p_5,p_6,p_7
A,2.0,2.0,1.5,1.518711604434893,0.1,2.400789101642099,2.400789101642099,2.400789101642099,0.06315476261390056
B,2.106525817516089,2.038672471649886,1.4752043120005407,1.518711604434893,0.1,2.400789101642099,2.400789101642099,2.400789101642099,0.06315476261390056
C,2.1651893082724047,2.1651893082724047,1.7902227199892276,1.9967121496392446,0.1,2.704712264824784,2.704712264824784,2.704712264824784,0.06315476261390056
"""
    df = read_csv_cv(IOBuffer(csv))
    DataFrames.transform!(df,
        :indiv_id => ByRow(Symbol) => :indiv_id,
    )
    df.u0[1][:sv₊x] .= [2.0, 2.0]
    df.p[1][:sv₊τ] = 1.5
    # testing some information missing in default, needed from prior
    keys_p_noi = setdiff(keys(df.p[1]), (:sv₊i,))
    DataFrames.transform(df,
        :p => ByRow(cv -> cv[keys_p_noi]) => :p)
    # testing no information for whole individuals
    subset!(df, :indiv_id => ByRow(≠(:C)))
    return(df)
end

# function get_case_u0p(::SampleSystemVecCase; scenario)
#     # creating the csv string:
#     # io = IOBuffer()
#     # CSV.write(io, indiv_info[:, 1:3])
#     # s = String(take!(io))
#     # print(s)
#     csv = """
# indiv_id,u0,p
# A,"(sv₊x = [2.0383292042153554, 2.0383292042153554])","(sv₊τ = 1.4009482259635606, sv₊i = 1.518711604434893, sv₊i2 = 0.1, sv₊p = [2.400789101642099, 2.400789101642099, 2.400789101642099])"
# B,"(sv₊x = [2.106525817516089, 2.038672471649886])","(sv₊τ = 1.4752043120005407, sv₊i = 1.518711604434893, sv₊i2 = 0.1, sv₊p = [2.400789101642099, 2.400789101642099, 2.400789101642099])"
# C,"(sv₊x = [2.010654503237803, 2.0510192980037196])","(sv₊τ = 1.4034321912259409, sv₊i = 1.518711604434893, sv₊i2 = 0.1, sv₊p = [2.400789101642099, 2.400789101642099, 2.400789101642099])"
# """
#     df = CSV.read(IOBuffer(csv), DataFrame)
#     DataFrames.transform!(df,
#         :indiv_id => ByRow(Symbol) => :indiv_id,
#         :u0 => ByRow(parse_nested_tuple) => :u0,
#         :p => ByRow(parse_nested_tuple) => :p)
#     df.u0[1][:sv₊x] .= [2.0, 2.0]
#     df.p[1][:sv₊τ] = 1.5
#     # testing some information missing in default, needed from prior
#     keys_p_noi = setdiff(keys(df.p[1]), (:sv₊i,))
#     DataFrames.transform(df,
#         :p => ByRow(cv -> cv[keys_p_noi]) => :p)
#     # testing no information for whole individuals
#     subset!(df, :indiv_id => ByRow(≠(:C)))
#     return(df)
# end
# function parse_nested_tuple(s)
#     # insert a comma before each closing bracket if comma is not there yet
#     # otherwise a single netry its not recognized as NamedTuple but as bracketed assignment
#     # negative lookbehind https://stackoverflow.com/a/9306228    
#     s1 = replace(s, r"(?<!,)\)" => ",)")
#     # replace by something less dangerous but more flexible than JLD - read_csv_cv
#     # such as storing numeric vectors - but then how to store the axes?
#     #    name [length?]
#     t = eval(Meta.parse(s1)) # NamedTuple
#     ComponentVector(t)
# end
