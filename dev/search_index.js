var documenterSearchIndex = {"docs":
[{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"CurrentModule = CrossInverts","category":"page"},{"location":"example_vec/#Walkthrough","page":"Walkthrough","title":"Walkthrough","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"This example demonstrates how  using an example system with symbolic array state and symbolic array parameters.","category":"page"},{"location":"example_vec/#Inversion-setup","page":"Walkthrough","title":"Inversion setup","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"The inversion setup is given by overriding specific functions with the first argument being a specific subtype of AbstractCrossInversionCase corresponding to the inversion problem.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Here, we define singleton type DocuVecCase and provide inversion setup  with defining methods for this type. Take care to add methods to the function in module CrossInverts rather define the methods in module Main.","category":"page"},{"location":"example_vec/#Example-system","page":"Walkthrough","title":"Example system","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"First lets setup the system to be inverted, using function get_case_inverted_system.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"using ModelingToolkit, OrdinaryDiffEq \nusing ModelingToolkit: t_nounits as t, D_nounits as D\nusing ComponentArrays: ComponentArrays as CA\nusing MTKHelpers\nusing CrossInverts\nusing DistributionFits\nusing PDMats: PDiagMat\nusing Turing\n\nfunction samplesystem_vec(; name, τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])\n    n_comp = 2\n    @parameters t\n    D = Differential(t)\n    @variables x(..)[1:n_comp] dec2(..) \n    ps = @parameters τ=τ i=i i2 p[1:3]=p\n    sts = vcat([x(t)[i] for i in 1:n_comp], dec2(t))\n    eq = [\n        D(x(t)[1]) ~ i - p[1] * x(t)[1] + (p[2] - x(t)[1]^2) / τ,\n        D(x(t)[2]) ~ i - dec2(t) + i2,\n        dec2(t) ~ p[3] * x(t)[2], # observable\n    ]\n    sys = ODESystem(eq, t, sts, vcat(ps...); name)\nend\n\nstruct DocuVecCase <: AbstractCrossInversionCase end\n\nfunction CrossInverts.get_case_inverted_system(::DocuVecCase; scenario)\n    @named sv = samplesystem_vec()\n    @named system = embed_system(sv)\n    u0_default = CA.ComponentVector() \n    p_default = CA.ComponentVector(sv₊i2 = 0.1)\n    (;system, u0_default, p_default)\nend\n\ninv_case = DocuVecCase()\nscenario = NTuple{0, Symbol}()\n(;system, u0_default, p_default) = get_case_inverted_system(inv_case; scenario)\nsystem","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Here, some parameters have default values, others, suche as sv₊i2, need to specified with with returned ComponentVector p_default.","category":"page"},{"location":"example_vec/#Optimized-parameters-and-individuals","page":"Walkthrough","title":"Optimized parameters and individuals","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"First, we define which parameters should be calibrated as fixed, ranmul, or individual parameters using function get_case_mixed_keys. Next, we define which individuals take part in the inversion scenario using function get_case_indiv_ids","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"function CrossInverts.get_case_mixed_keys(::AbstractCrossInversionCase; scenario)\n    (;\n        fixed = (:sv₊p,),\n        ranadd = (),\n        ranmul = (:sv₊x, :sv₊τ),\n        indiv = (:sv₊i,))\nend\n\nCrossInverts.get_case_indiv_ids(::DocuVecCase; scenario) = (:A, :B, :C)\nnothing # hide","category":"page"},{"location":"example_vec/#Priors,-Observations,-and-Observation-uncertainty","page":"Walkthrough","title":"Priors, Observations, and Observation uncertainty","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"We need to provide additional information to the inversion, such as observations, observation uncertainties, and prior distribution.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"We provide priors with function  get_case_priors_dict. For simplicity we return the same priors independent  of the individual or the scenario.  For the SymbolicArray parameters, we need to provide a Multivariate distribution. Here, we provide a product distribution of uncorrelated LogNormal distributions, which are specified by its mode and upper quantile using df_from_paramsModeUpperRows.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"function CrossInverts.get_case_priors_dict(::DocuVecCase, indiv_id; scenario = NTuple{0, Symbol}())\n    #using DataFrames, Tables, DistributionFits, Chain\n    paramsModeUpperRows = [\n        # τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])\n        (:sv₊i, LogNormal, 1.0, 6.0),\n        (:sv₊τ, LogNormal, 1.0, 5.0),\n        (:sv₊x_1, LogNormal, 1.0, 2.0),\n        (:sv₊x_2, LogNormal, 1.0, 2.0),\n    ]\n    df_scalars = df_from_paramsModeUpperRows(paramsModeUpperRows)\n    dd = Dict{Symbol, Distribution}(df_scalars.par .=> df_scalars.dist)\n    dist_p0 = fit(LogNormal, @qp_m(1.0), @qp_uu(3.0))\n    # dd[:sv₊p] = product_distribution(fill(dist_p0, 3))\n    # dd[:sv₊x] = product_distribution(dd[:sv₊x_1], dd[:sv₊x_2])\n    dd[:sv₊p] = product_MvLogNormal(fill(dist_p0, 3)...)\n    dd[:sv₊x] = product_MvLogNormal(dd[:sv₊x_1], dd[:sv₊x_2])\n    dd\nend\nfunction product_MvLogNormal(comp...)\n    μ = collect(getproperty.(comp, :μ))\n    σ = collect(getproperty.(comp, :σ))\n    Σ = PDiagMat(exp.(σ))\n    MvLogNormal(μ, Σ)\nend\n\nget_case_priors_dict(inv_case, :A; scenario)","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Similarly, we provide prior distributions for uncertainty of the random effects by function get_case_priors_random_dict.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"function CrossInverts.get_case_priors_random_dict(::DocuVecCase; scenario = NTuple{0, Symbol}())\n    # prior in σ rather than σstar\n    d_exp = Exponential(log(1.05))\n    dd = Dict{Symbol, Distribution}([:sv₊τ, :sv₊i] .=> d_exp)\n    dd[:sv₊x] = Distributions.Product(fill(d_exp, 2))\n    dd\nend\n\nget_case_priors_random_dict(inv_case; scenario)","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Further, the type of distribution of observation uncertainties of  the observations of different data streams by function  get_case_obs_uncertainty_dist_type.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"function CrossInverts.get_case_obs_uncertainty_dist_type(::DocuVecCase, stream;\n        scenario = NTuple{0, Symbol}())\n    dtypes = Dict{Symbol, Type}(:sv₊dec2 => LogNormal,\n        :sv₊x => MvLogNormal)\n    dtypes[stream]\nend\n\nget_case_obs_uncertainty_dist_type(inv_case, :sv₊dec2; scenario)","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Finally, for each","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"each individual, \nfor each stream, ","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"we provide a vectors of","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"t: time\nobs: observations (vectors for multivariate variables)\nobs_unc: observation uncertainty parameters (can be matrices for multivariate variables)\nobs_true (optionally): values of the true model to be rediscovered in synthetic experiments","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"This is done by implementing function get_case_indivdata. Usually, this would be read information from a file or database. Here, we provide  the numbers as text.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"function CrossInverts.get_case_indivdata(::DocuVecCase, indiv_id; scenario = NTuple{0, Symbol}())\n    data = (A = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],\n                obs = [\n                    [2.3696993004601956, 2.673733320916141],\n                    [1.8642844249865063, 2.0994355527637607],\n                    [1.9744553950945931, 2.049494086682751],\n                    [1.806115091024414, 1.4088107777562726],\n                ],\n                obs_unc = [\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                ],\n                obs_true = [\n                    [1.4528506430586314, 1.502300054146255],\n                    [1.2174085538439976, 1.1706665606844529],\n                    [1.0483430119731987, 0.7600115428483291],\n                    [1.0309694961068738, 0.6441417808271487],\n                ]),\n            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],\n                obs = [\n                    3.7951565919532038,\n                    2.932295276687423,\n                    2.0064853619502925,\n                    1.6522510350996853,\n                ],\n                obs_unc = [1.1, 1.1, 1.1, 1.1],\n                obs_true = [\n                    3.606705597390664,\n                    2.810523520548073,\n                    1.8246274291924653,\n                    1.546448567322152,\n                ])),\n        B = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],\n                obs = [\n                    [2.0681893973690264, 2.76555266499398],\n                    [3.002213659926257, 2.738988031384357],\n                    [2.2024778579768736, 1.8863521088263966],\n                    [1.8970493973645883, 1.4592874111525584],\n                ],\n                obs_unc = [\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                ],\n                obs_true = [\n                    [1.4319499386364825, 1.4846599446224278],\n                    [1.2097697867481565, 1.1597529395039063],\n                    [1.0512489486634184, 0.7574273823278419],\n                    [1.035264629162679, 0.6439076211840167],\n                ]),\n            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],\n                obs = [\n                    5.286801850397016,\n                    2.9649984441621826,\n                    2.1180756620394585,\n                    2.6749483017364,\n                ],\n                obs_unc = [1.1, 1.1, 1.1, 1.1],\n                obs_true = [\n                    3.5643554146940866,\n                    2.784322217758367,\n                    1.8184234047779861,\n                    1.5458863994028762,\n                ])),\n        C = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],\n                obs = [\n                    [2.2350643301157382, 2.3130035358019856],\n                    [2.0736166580761624, 1.9436035468232888],\n                    [2.0472448291872816, 1.529804596360485],\n                    [1.8267544248914431, 1.2760177129115113],\n                ],\n                obs_unc = [\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],\n                ],\n                obs_true = [\n                    [1.4810168420659708, 1.502512426277095],\n                    [1.226148237932659, 1.1707979724544357],\n                    [1.0387515337959667, 0.7600427779041109],\n                    [1.0183823891718273, 0.6441445598911335],\n                ]),\n            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],\n                obs = [\n                    4.026668907719985,\n                    3.1937462073315097,\n                    6.2700505882164785,\n                    3.4322758342125548,\n                ],\n                obs_unc = [1.1, 1.1, 1.1, 1.1],\n                obs_true = [\n                    3.607215458087877,\n                    2.8108390124932754,\n                    1.8247024179739757,\n                    1.5464552392686794,\n                ])))\n    data[indiv_id]\nend\n\nget_case_indivdata(inv_case, :A; scenario)","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Often, when one parameter is adjusted, this has consequences for other non-optimized parameters. Function get_case_problemupdater allows to provide a ParameterUpdater  to take care. In this example, when optimizing parameter i, then parameter i2 is set to the same value.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"function CrossInverts.get_case_problemupdater(::DocuVecCase; system, scenario = NTuple{0, Symbol}())\n    mapping = (:sv₊i => :sv₊i2,)\n    pset = ODEProblemParSetter(system, Symbol[]) # parsetter to get state symbols\n    get_ode_problemupdater(KeysProblemParGetter(mapping, keys(axis_state(pset))), system)\nend\n\nget_case_problemupdater(inv_case; system, scenario)","category":"page"},{"location":"example_vec/#Compiling-the-setup","page":"Walkthrough","title":"Compiling the setup","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"With all the functions of the setup defined, we can call function  setup_inversion to compile all the setup. We get the system object, information at population level, and information at individual level as a DataFrame.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"(;system, indiv_info, pop_info) = setup_inversion(inv_case; scenario)\nkeys(pop_info.sample0)","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"A single sample is a ComponentVector with components","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"fixed: fixed effects\nranadd: mean additive random effects\nranmul: mean multiplicative random effects\nranadd_σ: uncertainty parameter of the additive random effects\nranmul_σ: uncertainty parameter of the multiplicative random effects\nindiv: Component vector of each individual with individual effects\nindiv_ranadd: Difference between individual and mean additive random effect\nindiv_ranmul: Ratio betweenn individual and mean multiplicative random effect","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"A reminder of the effects:","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"pop_info.mixed_keys","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Accessing single components.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"pop_info.sample0[:ranmul]","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"pop_info.sample0[:indiv][:A]","category":"page"},{"location":"example_vec/#Forward-simulation","page":"Walkthrough","title":"Forward simulation","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Although not necessary for the inversion, it can be helpful for  analysing to do a single forward simulation for all individuals  for a given estimate of the effects.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"First, a function is created using gen_sim_sols_probs  that requires an estimate of the effects, and returns the solution and the updated problem for each individual. Then this function is called with initial estimates.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"solver = AutoTsit5(Rodas5P())\nsim_sols_probs = gen_sim_sols_probs(; \n    tools = indiv_info.tools, psets = pop_info.psets, \n    problemupdater = pop_info.problemupdater, solver)\n(;fixed, ranadd, ranmul, indiv, indiv_ranadd, indiv_ranmul) = pop_info.mixed\nsols_probs = sim_sols_probs(fixed, ranadd, ranmul, indiv, indiv_ranadd, indiv_ranmul)\n(sol, problem_opt) = sols_probs[1]\nsol[:sv₊x]\nnothing # hide","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Below we just check that the ProblemUpdater really updated the non-optimized parameter i2 to the value of the optimized parameter i.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"pset = pop_info.psets.fixed\nget_par_labeled(pset, problem_opt)[:sv₊i2] == get_par_labeled(pset, problem_opt)[:sv₊i]","category":"page"},{"location":"example_vec/#Model-Inversion","page":"Walkthrough","title":"Model Inversion","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"First, a Turing-model is created using gen_model_cross. Next, a few samples are drawn from this model using the NUTS sampler.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"model_cross = gen_model_cross(;\n    inv_case, tools = indiv_info.tools, \n    priors_pop = pop_info.priors_pop, psets = pop_info.psets, \n    sim_sols_probs, scenario, solver);\n\nn_burnin = 0\nn_sample = 10\nchn = Turing.sample(model_cross, Turing.NUTS(n_burnin, 0.65, init_ϵ = 0.2), n_sample,\n    init_params = collect(pop_info.sample0))\n\nnames(chn, :parameters)","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"For each scalarized value of the effects there is a series of samples.","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"a group estimate for each fixed effect. For multivariate variables the index is appended last, e.g. Symbol(\"fixed[:sv₊p][1]\").\na group mean additive random effect (none in the example case).\na group mean multiplicative random effect, e.g. Symbol(\"ranmul[:sv₊τ]\").\nan uncertainty parameter of the ranmul effect, e.g. Symbol(\"pranmul_σ[:sv₊τ]\").\nan individual effect for each individual, e.g. Symbol(\"indiv[:sv₊i, 3]\")  for the third individual.\nthe individual offset for the ranadd effect for each individual (none in the example case),\nthe individual multiplier for the ranmul effect for each individual, e.g. Symbol(\"indiv_ranmul[:sv₊τ, 3]\").","category":"page"},{"location":"example_vec/#Extracting-individual-effects","page":"Walkthrough","title":"Extracting individual effects","text":"","category":"section"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"Each row of a multivariate chain can be extracted as a ComponentVector as described in [Extracting effects from sampled object].","category":"page"},{"location":"example_vec/","page":"Walkthrough","title":"Walkthrough","text":"chn2 = chn[:,vcat(pop_info.effect_pos[:indiv_ranmul][:B]...),:]\nchn3 = extract_group(chn2, :indiv_ranmul, pop_info.indiv_ids)\nnames(chn3)","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"CurrentModule = CrossInverts","category":"page"},{"location":"extract_groups/#Extracting-effects-from-sampled-object","page":"Extracting effects","title":"Extracting effects from sampled object","text":"","category":"section"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"When setting up the inversion using setup_inversion, the ComponentVector pop_info.effect_pos is returned, which specifies the positions of effects in each sample.","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"First lets mockup a sampling result and corresponding effect positions.","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"using CrossInverts\nusing MCMCChains\nusing ComponentArrays: ComponentArrays as CA\n\n\nchn = Chains(randn((10,21,2)), [\"fixed[:p][1]\", \"fixed[:p][2]\", \"fixed[:p][3]\", \"ranmul[:x][1]\", \"ranmul[:x][2]\", \"ranmul[:τ]\", \"pranmul_σ[:x][1]\", \"pranmul_σ[:x][2]\", \"pranmul_σ[:τ]\", \"indiv[:i, 1]\", \"indiv[:i, 2]\", \"indiv[:i, 3]\", \"indiv_ranmul[:x, 1][1]\", \"indiv_ranmul[:x, 1][2]\", \"indiv_ranmul[:τ, 1]\", \"indiv_ranmul[:x, 2][1]\", \"indiv_ranmul[:x, 2][2]\", \"indiv_ranmul[:τ, 2]\", \"indiv_ranmul[:x, 3][1]\", \"indiv_ranmul[:x, 3][2]\", \"indiv_ranmul[:τ, 3]\"])\n\npop_info = (;\n    effect_pos = CA.ComponentVector(fixed = (p = 1:3,), ranmul = (x = 4:5, τ = 6), \n        ranmul_σ = (x = 7:8, τ = 9), indiv = (A = (i = 10), B = (i = 11), \n        C = (i = 12)), indiv_ranmul = (A = (x = 13:14, τ = 15), \n        B = (x = 16:17, τ = 18), C = (x = 19:20, τ = 21))),\n    indiv_ids = (:A, :B, :C))\nnothing # hide","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"The effect_pos can be conveniently used to index into a sample, e.g. to extract sub-Chains from the sampling results. Lets get the ranmul multipliers for individual :B, the vector parameter :x and scalar τ.","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"pop_info.effect_pos[:indiv_ranmul][:B]","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"chn2 = chn[:,vcat(pop_info.effect_pos[:indiv_ranmul][:B]...),:]\nnames(chn2)","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"The names can be simplified by constraining to a single group using extract_group. In addition, this allows replacing the indices of  individuals by more readable identifiers.","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"chn3 = extract_group(chn2, :indiv_ranmul, pop_info.indiv_ids)\nnames(chn3)","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"Alternatively, one can attach the ComponentArrays-Axis  to the array constructed from the chain and index into it.","category":"page"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"s1 = CA.ComponentMatrix(Array(chn),\n    CA.FlatAxis(), first(CA.getaxes(pop_info.effect_pos)))\n# sv.p within fixed\ns1[:, :fixed][:, :p]\n# ranmul effects multiplier for site B for ranmul parameter tau\ns1[:, :indiv_ranmul][:, :B][:, :τ]\nnothing # hide","category":"page"},{"location":"extract_groups/#API","page":"Extracting effects","title":"API","text":"","category":"section"},{"location":"extract_groups/","page":"Extracting effects","title":"Extracting effects","text":"extract_group","category":"page"},{"location":"extract_groups/#CrossInverts.extract_group","page":"Extracting effects","title":"CrossInverts.extract_group","text":"extract_group(chn, group::Symbol) \nextract_group(chn, group::Symbol, indiv_ids)\n\nExtract SubChain of components  (:fixed, :ranmul, :ranadd, :pranmul_σ, :pranadd_σ, :indiv, :indiv_ranmul, :indiv_ranadd) and replace parameter names like group[:p1, o1]o2 => group[o1]o2. The variant, where indiv_ids is specified, additionally  replaces first [i] by [:(indiv_ids[i])].\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = CrossInverts","category":"page"},{"location":"#CrossInverts","page":"Home","title":"CrossInverts","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for CrossInverts.","category":"page"},{"location":"#Problem","page":"Home","title":"Problem","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"When calibrating models across different groups, some parameters should be kept constant across sites (fixed parameters), some are allowed to vary  around a common mean across sites (random parameters), and others may be allowed to vary freely across groups (individual parameters).","category":"page"},{"location":"","page":"Home","title":"Home","text":"Specifying priors and and dealing with groups and individuals require some common support. Further, keeping the overview across different scenarios can be tedious.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"inversion_case/","page":"Providing inversion information","title":"Providing inversion information","text":"CurrentModule = CrossInverts","category":"page"},{"location":"inversion_case/#Providing-inversion-information","page":"Providing inversion information","title":"Providing inversion information","text":"","category":"section"},{"location":"inversion_case/","page":"Providing inversion information","title":"Providing inversion information","text":"AbstractCrossInversionCase\nget_case_inverted_system\nget_case_mixed_keys\nget_case_indiv_ids\nget_case_priors_dict\nget_case_priors_random_dict\nget_case_obs_uncertainty_dist_type\nget_case_indivdata\nget_case_problemupdater","category":"page"},{"location":"inversion_case/#CrossInverts.AbstractCrossInversionCase","page":"Providing inversion information","title":"CrossInverts.AbstractCrossInversionCase","text":"AbstractCrossInversionCase\n\nInterface for providing all relevant information for a cross-individual mixed effects bayesian inversion.\n\nConcrete types should implement\n\nget_case_inverted_system(::AbstractCrossInversionCase; scenario) \nget_case_mixed_keys(::AbstractCrossInversionCase; scenario)\nget_case_indiv_ids(::AbstractCrossInversionCase; scenario)\nget_case_priors_dict(::AbstractCrossInversionCase, indiv_id; scenario) Priors for optimized model effects, which may differ by individual. \nget_case_priors_random_dict(::AbstractCrossInversionCase; scenario) Priors for meta-parameters for ranadd and ranmul effects.\nget_case_obs_uncertainty_dist_type(::AbstractCrossInversionCase; scenario) Type of distribution of observation-uncertainty per stream.\nget_case_indivdata(::AbstractCrossInversionCase, indiv_id; scenario) The times, observations, and uncertainty parameters per indiv_id and stream.\n\noptionally:\n\nget_case_u0p(::AbstractCrossInversionCase; scenario)  \nget_case_problemupdater(::AbstractCrossInversionCase; system, scenario) A ProblemUpdater for ensuring consistent parameters after setting optimized  parameters.\n\n\n\n\n\n","category":"type"},{"location":"inversion_case/#CrossInverts.get_case_inverted_system","page":"Providing inversion information","title":"CrossInverts.get_case_inverted_system","text":"get_case_inverted_system(::AbstractCrossInversionCase; scenario)\n\nProvide a NamedTuple  (;system::AbstractSystem, u0_default::ComponentVector, p_default::ComponentVector) of the inverted system and default initial values and parameters  for given inversion scenario.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.get_case_mixed_keys","page":"Providing inversion information","title":"CrossInverts.get_case_mixed_keys","text":"get_case_mixed_keys(::AbstractCrossInversionCase; scenario)\n\nProvide NamedTuple (;fixed, ranadd, ranmul, indiv) of tuples of parameter names (Symbol) that are optimized in the inversion scenario.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.get_case_indiv_ids","page":"Providing inversion information","title":"CrossInverts.get_case_indiv_ids","text":"get_case_indiv_ids(::AbstractCrossInversionCase; scenario)\n\nProvide Tuple of Symbols identifying the individuals.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.get_case_priors_dict","page":"Providing inversion information","title":"CrossInverts.get_case_priors_dict","text":"get_case_priors_dict(::AbstractCrossInversionCase, indiv_id; scenario)\n\nProvide a dictionary (par -> Distribution) for prior parameters and unknowns.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.get_case_priors_random_dict","page":"Providing inversion information","title":"CrossInverts.get_case_priors_random_dict","text":"get_case_priors_random_dict(::AbstractCrossInversionCase; scenario)\n\nProvide a dictionary (par -> Distribution) for hyperpriors of the spread of the ranmul and ranadd effects.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.get_case_obs_uncertainty_dist_type","page":"Providing inversion information","title":"CrossInverts.get_case_obs_uncertainty_dist_type","text":"get_case_obs_uncertainty_dist_type(::AbstractCrossInversionCase; scenario)\n\nProvide the type of distribution of observation uncertainty for given stream, to be used with fit_mean_Σ.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.get_case_indivdata","page":"Providing inversion information","title":"CrossInverts.get_case_indivdata","text":"get_case_indivdata(::AbstractCrossInversionCase, indiv_id; scenario)\n\nProvide Tuple (indiv_id -> (stream_info) for each individ. Where StreamInfo is a Tuple `(streamsymbol -> (;t, obs, obstrue)). Such that solution can be indexed by sol[streamsymbol](t) to provide observations. Valueobs_true` is optional. They are synthetic data without noise  generated from the system, which are not used in inversion, but used for comparison.\n\nThe ValueType dispatches to different implementations. There is  am implementation for Val(:CrossInverts_samplesystem1) independent of scenario.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.get_case_problemupdater","page":"Providing inversion information","title":"CrossInverts.get_case_problemupdater","text":"get_case_problemupdater(::AbstractCrossInversionCase; scenario)\n\nReturn a specific ProblemUpdater for given Inversioncase and scennario. It is applied after parameters to optimized have been set. The typical case is optimizing a parameter, but adjusting other non-optimized parameters to be consistent with the optimized one, e.g. always use the same value for another parameter.\n\nThe default is a NullProblemUpdater, which does not modify parameters.    \n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#Setup-tools","page":"Providing inversion information","title":"Setup tools","text":"","category":"section"},{"location":"inversion_case/","page":"Providing inversion information","title":"Providing inversion information","text":"setup_inversion\ndf_from_paramsModeUpperRows","category":"page"},{"location":"inversion_case/#CrossInverts.setup_inversion","page":"Providing inversion information","title":"CrossInverts.setup_inversion","text":"setup_inversion(inv_case::AbstractCrossInversionCase; scenario = NTuple{0, Symbol}())\n\nCalls all the functions for specific AbstractCrossInversionCase to setup the inversion.\n\nReturns a NamedTuple with entries: (; system, pop_info, indiv_info)\n\npop_info is a NamedTuple with entriees\n\nmixed_keys: The optimized parameters and their distribution across individual  as returned by [get_case_mixed_keys](@ref)\nindiv_ids: A tuple of ids (Symbols) of the individuals taking part in the   inversion, as returned by [get_case_indiv_ids](@ref)\nmixed: mixed effects NamedTuple(fixed, ranadd, ranmul, indiv, indiv_ranadd, indiv_ranmul) from individual's states and parameters in the format expected by forward sim.\npsets: NTuple{ODEProblemParSetter} for each mixed component\npriors_pop: ComponentVector of priors on population level  (fixed, ranadd, ranmul, ranadd_σ, ranmul_σ)\nproblemupdaterproblemupdater: used to update problems after setting parameters to be optimized.\nsample0: ComponentVector of an initial sample This can be used to name (attach_axis) a sample from MCMCChains object \neffect_pos: ComponentVector to map keys to positions (1:n_par) in sample\n\nindiv_info: is a DataFrame with rows for each individual with  columns\n\nindiv_id: Symbols identifying individuals\nu0 and p: ComponentVectors of initial states and parameters\nparopt: optimized parameters extracted from indiviudals state and parameters\ntools: tools initialized for each site \npriors_indiv: priors, which may differ between individuals\nproblem: the ODEProblem set up with initial state and parameters\nindivdata: as returned by getcaseindivdata\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.df_from_paramsModeUpperRows","page":"Providing inversion information","title":"CrossInverts.df_from_paramsModeUpperRows","text":"df_from_paramsModeUpperRows(paramsModeUpperRows)\n\nConvert Tuple-Rows of (:par, :dType, :mode, :upper) to DataFrame. And fit distribution and report it in column :dist.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#Simulating","page":"Providing inversion information","title":"Simulating","text":"","category":"section"},{"location":"inversion_case/","page":"Providing inversion information","title":"Providing inversion information","text":"gen_sim_sols_probs\ngen_model_cross","category":"page"},{"location":"inversion_case/#CrossInverts.gen_sim_sols_probs","page":"Providing inversion information","title":"CrossInverts.gen_sim_sols_probs","text":"gen_sim_sols_probs(; tools, psets, problemupdater, \n    solver = AutoTsit5(Rodas5()), kwargs_gen...)\n\nGenerates a function sim_sols_probs(fixed, ranadd, ranmul, indiv, indiv_ranmul, indiv_ranadd) that updates and simulate the system (given with tools to gen_sim_sols_probs) by \n\nfor each individual i\nupdate fixed parameters: fixed \nupdate ranadd parameters: ranadd .+ indiv_ranadd[:,i]\nupdate ranmul parameters: ranmul .* indiv_ranmul[:,i]\nupdate indiv_id parameters: indiv[:,i]\nsimulate the problem\nreturn a vector(nindiv) of (;sol, problemopt)\n\nIf non-optimized p or u0 differ between individuals, they must already be set in tools[i_indiv].problem.\n\n\n\n\n\n","category":"function"},{"location":"inversion_case/#CrossInverts.gen_model_cross","page":"Providing inversion information","title":"CrossInverts.gen_model_cross","text":"function gen_model_cross(;\n        inv_case::AbstractCrossInversionCase, tools, priors_pop, sim_sols_probs,\n        scenario, psets, solver)\n\nGenerate a Turing model using objects generated by setup_inversion  components: indiv_info.tools, pop_info.priors_pop, and pop_info.psets) and forward simulation function generated by gen_sim_sols_probs.\n\n\n\n\n\n","category":"function"}]
}
