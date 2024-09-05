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
    df_scalars = fit_dists_mode_upper(paramsModeUpperRows)
    dd = Dict{Symbol, Distribution}(df_scalars.par .=> df_scalars.dist)
    dist_p0 = fit(LogNormal, 1.0, @qp_uu(3.0), Val(:mode))
    # dd[:sv₊p] = product_distribution(fill(dist_p0, 3))
    # dd[:sv₊x] = product_distribution(dd[:sv₊x_1], dd[:sv₊x_2])
    dd[:sv₊p] = product_MvLogNormal(fill(dist_p0, 3)...)
    dd[:sv₊x] = product_MvLogNormal(dd[:sv₊x_1], dd[:sv₊x_2])
    return (dd)
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
    return (dd)
end

function get_case_obs_uncertainty_dist_type(::SampleSystemVecCase, stream;
        scenario = NTuple{0, Symbol}())
    dtypes = Dict{Symbol, Type}(:sv₊dec2 => LogNormal,
        :sv₊x => MvLogNormal, :sv₊b1obs => Normal)
    dtypes[stream]
end

function get_case_indivdata(::SampleSystemVecCase, indiv_id; scenario = NTuple{0, Symbol}())
    # generated as in test_util_mixed.jl testset simulate_indivdata
    data = (
        A = (
            sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [[1.9066941798921013, 1.938406312968056],
                    [1.12022207456593, 1.0758155239871716],
                    [1.3723063737558432, 0.5781072294078848],
                    [0.8173564987064316, 0.8044329888539079]],
                obs_unc = [[0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [[1.6562250691126008, 1.7170376974381363],
                    [1.2998861385114096, 1.3194042055281718],
                    [1.0426271099028963, 0.8270254999637731],
                    [1.0145112821451405, 0.6880948021231954]]),
            sv₊b1obs = (t = [0.1], obs = [-0.31686954026973096],
                obs_unc = [0.5], obs_true = [0.06315476261390056]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [3.9028257395901105, 3.0405965225455707,
                    1.8523986002610147, 1.6732202374355705],
                obs_unc = [0.09531017980432493, 0.09531017980432493,
                    0.09531017980432493, 0.09531017980432493],
                obs_true = [4.122245391118121, 3.1676112372927867,
                    1.9855138070931344, 1.6519705018339441])),
        B = (
            sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [[1.297828839669444, 1.358296145170648],
                    [0.8296166415159397, 1.148094168761115],
                    [0.9044764384836562, 1.4111333603248684],
                    [1.3767931719574817, 1.3092492346986238]],
                obs_unc = [[0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [[1.6860135285379814, 1.6802859143457225],
                    [1.312948077553763, 1.2966664200132834],
                    [1.047995257932727, 0.8216414908906547],
                    [1.0197681667919476, 0.687606191490374]]),
            sv₊b1obs = (t = [0.1], obs = [0.37937432656723447],
                obs_unc = [0.5], obs_true = [0.06315476261390056]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [4.320055797981225, 3.604753454395906,
                    1.672256629238385, 1.6059790201330124],
                obs_unc = [0.09531017980432493, 0.09531017980432493,
                    0.09531017980432493, 0.09531017980432493],
                obs_true = [4.03401211080394, 3.113022609633167,
                    1.9725879367872496, 1.65079745075172])),
        C = (
            sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [[1.146796377211607, 1.5135383983184902],
                    [1.708062650525056, 1.7043425162970816],
                    [1.2175800962239545, 1.2494889366008093],
                    [0.3945481210384016, 1.217719072057798]],
                obs_unc = [[0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493]],
                obs_true = [[1.7989311286283927, 1.9820954072646162],
                    [1.382640691624001, 1.4778557902392064],
                    [1.0920265658761596, 0.9138691659965111],
                    [1.0627178605569891, 0.7844860789908807]]),
            sv₊b1obs = (t = [0.1], obs = [0.3229569198504482],
                obs_unc = [0.5], obs_true = [0.06315476261390056]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [5.179317680240015, 4.572734161990259,
                    2.7603999980014375, 2.144214138519791],
                obs_unc = [0.09531017980432493, 0.09531017980432493,
                    0.09531017980432493, 0.09531017980432493],
                obs_true = [5.360997758081483, 3.997174681502305,
                    2.4717531417159604, 2.1218091194309396])))
    data[indiv_id]
end

function get_case_problemupdater(
        ::SampleSystemVecCase; system, scenario = NTuple{0, Symbol}())
    mapping = (:sv₊i => :sv₊i2,)
    pset = ODEProblemParSetter(system, Symbol[]) # parsetter to get state symbols
    get_ode_problemupdater(KeysProblemParGetter(mapping, keys(axis_state(pset))), system)
end

function get_case_u0p(::SampleSystemVecCase; scenario)
    :no_u0p ∈ scenario && return (DataFrame(indiv_id = Symbol[]))
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
        :indiv_id => ByRow(Symbol) => :indiv_id
    )
    df.u0[1][:sv₊x] .= [2.0, 2.0]
    df.p[1][:sv₊τ] = 1.5
    # testing some information missing in default, needed from prior
    keys_p_noi = setdiff(keys(df.p[1]), (:sv₊i,))
    DataFrames.transform(df,
        :p => ByRow(cv -> cv[keys_p_noi]) => :p)
    # testing no information for whole individuals
    subset!(df, :indiv_id => ByRow(≠(:C)))
    return (df)
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
