```@meta
CurrentModule = CrossInverts
```

# Walkthrough

This example demonstrates how 
using an example system with symbolic array state and symbolic array parameters.

## Example system

```@example doc
using ModelingToolkit, OrdinaryDiffEq 
using ModelingToolkit: t_nounits as t, D_nounits as D
using ComponentArrays: ComponentArrays as CA
using MTKHelpers
using CrossInverts
using DistributionFits
using PDMats: PDiagMat
using Turing

function samplesystem_vec(; name, τ = 3.0, i = 0.1, p = [1.1, 1.2, 1.3])
    n_comp = 2
    @parameters t
    D = Differential(t)
    @variables x(..)[1:n_comp] dec2(..) #dx(t)[1:2]  # observed dx now can be accessed
    ps = @parameters τ=τ i=i p[1:3]=p
    sts = vcat([x(t)[i] for i in 1:n_comp], dec2(t))
    eq = [
        D(x(t)[1]) ~ i - p[1] * x(t)[1] + (p[2] - x(t)[1]^2) / τ,
        D(x(t)[2]) ~ i - dec2(t),
        dec2(t) ~ p[3] * x(t)[2], # observable
    ]
    sys = ODESystem(eq, t, sts, vcat(ps...); name)
end

@named sv = samplesystem_vec()
@named system = embed_system(sv)
```


First, we define which parameters should be calibrated as fixed, random, or
individual parameters. Note, that we can use symbols rather than Symbolics,
and can use symbolic arrarys rather than requiring the scalarized version.
```@example doc
mixed_keys = (;
    fixed = (:sv₊p,),
    random = (:sv₊x, :sv₊τ),
    indiv = (:sv₊i,),)

indiv_ids = (:A, :B, :C)
nothing # hide
```

### Priors, Observations, and Observation uncertainty

We need to provide additional information to the inversion, such as observations,
observation uncertainties, and prior distribution.
This is achieved by overriding specific functions with the first argument being
a specific subtype of [AbstractCrossInversionCase](@ref) corresponding to the inversion problem.

Here, we define singleton type `DocuVecCase` and provide priors with function 
[`get_priors_dict`](@ref). For simplicity we return the same priors independent of the individual
or the scenario.
For the SymbolicArray parameters, we need to provide a Multivariate distribution.
Here, we provide a product distribution of uncorrelated LogNormal distributions,
which are specified by its mode and upper quantile using [`df_from_paramsModeUpperRows`](@ref).

```@example doc
struct DocuVecCase <: AbstractCrossInversionCase end

function CrossInverts.get_priors_dict(::DocuVecCase, indiv_id; scenario = NTuple{0, Symbol}())
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
function product_MvLogNormal(comp...)
    μ = collect(getproperty.(comp, :μ))
    σ = collect(getproperty.(comp, :σ))
    Σ = PDiagMat(exp.(σ))
    MvLogNormal(μ, Σ)
end

inv_case = DocuVecCase()
scenario = NTuple{0, Symbol}()

get_priors_dict(inv_case, :A; scenario)
```

Similarly, we provide prior distributions for uncertainty of
the random effects by function [`get_priors_random_dict`](@ref).

```@example doc
function CrossInverts.get_priors_random_dict(::DocuVecCase; scenario = NTuple{0, Symbol}())
    # prior in σ rather than σstar
    d_exp = Exponential(log(1.05))
    dd = Dict{Symbol, Distribution}([:sv₊τ, :sv₊i] .=> d_exp)
    dd[:sv₊x] = Distributions.Product(fill(d_exp, 2))
    dd
end

get_priors_random_dict(inv_case; scenario)
```

Further, the type of distribution of observation uncertainties of 
the observations of different data streams by function 
[`get_obs_uncertainty_dist_type`](@ref).

```@example doc
function CrossInverts.get_obs_uncertainty_dist_type(::DocuVecCase, stream;
        scenario = NTuple{0, Symbol}())
    dtypes = Dict{Symbol, Type}(:sv₊dec2 => LogNormal,
        :sv₊x => MvLogNormal)
    dtypes[stream]
end

get_obs_uncertainty_dist_type(inv_case, :sv₊dec2; scenario)
```

Finally, for each
- each individual, 
- for each stream, 
we provide a vectors of
- t: time
- obs: observations (vectors for multivariate variables)
- obs_unc: observation uncertainty parameters (can be matrices for multivariate variables)
- obs_true (optionally): values of the true model to be rediscovered
  in synthetic experiments

This is done by implementing function [`get_indivdata`](@ref).
Usually, this would be read information from a file or database. Here, we provide 
the numbers as text.

```@example doc   
function CrossInverts.get_indivdata(::DocuVecCase, indiv_id; scenario = NTuple{0, Symbol}())
    data = (A = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [2.3696993004601956, 2.673733320916141],
                    [1.8642844249865063, 2.0994355527637607],
                    [1.9744553950945931, 2.049494086682751],
                    [1.806115091024414, 1.4088107777562726],
                ],
                obs_unc = [
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                ],
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
                    [2.0681893973690264, 2.76555266499398],
                    [3.002213659926257, 2.738988031384357],
                    [2.2024778579768736, 1.8863521088263966],
                    [1.8970493973645883, 1.4592874111525584],
                ],
                obs_unc = [
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                ],
                obs_true = [
                    [1.4319499386364825, 1.4846599446224278],
                    [1.2097697867481565, 1.1597529395039063],
                    [1.0512489486634184, 0.7574273823278419],
                    [1.035264629162679, 0.6439076211840167],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    5.286801850397016,
                    2.9649984441621826,
                    2.1180756620394585,
                    2.6749483017364,
                ],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [
                    3.5643554146940866,
                    2.784322217758367,
                    1.8184234047779861,
                    1.5458863994028762,
                ])),
        C = (sv₊x = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    [2.2350643301157382, 2.3130035358019856],
                    [2.0736166580761624, 1.9436035468232888],
                    [2.0472448291872816, 1.529804596360485],
                    [1.8267544248914431, 1.2760177129115113],
                ],
                obs_unc = [
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                    [0.09531017980432493 0.0; 0.0 0.09531017980432493],
                ],
                obs_true = [
                    [1.4810168420659708, 1.502512426277095],
                    [1.226148237932659, 1.1707979724544357],
                    [1.0387515337959667, 0.7600427779041109],
                    [1.0183823891718273, 0.6441445598911335],
                ]),
            sv₊dec2 = (t = [0.2, 0.4, 1.0, 2.0],
                obs = [
                    4.026668907719985,
                    3.1937462073315097,
                    6.2700505882164785,
                    3.4322758342125548,
                ],
                obs_unc = [1.1, 1.1, 1.1, 1.1],
                obs_true = [
                    3.607215458087877,
                    2.8108390124932754,
                    1.8247024179739757,
                    1.5464552392686794,
                ])))
    data[indiv_id]
end

get_indivdata(inv_case, :A; scenario)
```

## Extracting initial information and tools

A first estimate of the optimized initial state and parameters can then be obtained
from priors using function [`get_indiv_parameters_from_priors`](@ref), 
and a set of tools is created using function [`setup_tools_mixed`](@ref)

```@example doc
p_indiv = get_indiv_parameters_from_priors(inv_case; 
    scenario, indiv_ids, mixed_keys, system)
(;mixed, indiv_info, pop_info) = setup_tools_mixed(p_indiv;
    inv_case, scenario, system, mixed_keys)
#(psets, priors_pop, sample0, effect_pos) = pop_info
keys(pop_info.sample0)
```

A single sample is a ComponentVector with components
- fixed: fixed effects
- random: mean random effects
- random_σ: uncertainty parameter of the random effects
- indiv: Component vector of each site with individual effects
- indiv_random: 

A reminder of the effects:
```@example doc
mixed_keys
```

Accessing single components.
```@example doc
pop_info.sample0[:random]
```
```@example doc
pop_info.sample0[:indiv][:A]
```

## Forward simulation

Although not necessary for the inversion, it can be helpful for 
analysing to do a single forward simulation for all individuals 
for a given estimate of the effects.

First, a function is created using [`gen_sim_sols_probs`](@ref) 
that requires an estimate of the effects,
and returns the solution and the updated problem for each individual.
Then this function is called with initial estimates.

```@example doc
solver = AutoTsit5(Rodas5P())
sim_sols_probs = gen_sim_sols_probs(; 
    tools = indiv_info.tools, psets = pop_info.psets, 
    problemupdater = pop_info.problemupdater, solver)
(fixed, random, indiv, indiv_random) = mixed
sols_probs = sim_sols_probs(fixed, random, indiv, indiv_random)
(sol, problem_opt) = sols_probs[1]
sol[:sv₊x]
nothing # hide
```

## Model Inversion

First, a Turing-model is created using [`gen_model_cross`](@ref).
Next, a few samples are drawn from this model using the NUTS sampler.

```@example doc
model_cross = gen_model_cross(;
    inv_case, tools = indiv_info.tools, 
    priors_pop = pop_info.priors_pop, psets = pop_info.psets, 
    sim_sols_probs, scenario, solver);

n_burnin = 0
n_sample = 10
chn = Turing.sample(model_cross, Turing.NUTS(n_burnin, 0.65, init_ϵ = 0.2), n_sample,
    init_params = collect(pop_info.sample0))

names(chn, :parameters)
```

For each scalarized value of the effects there is a series of samples.
- a single estimate for each fixed effect. For multivariate variables
the index is appended last, e.g. `Symbol("fixed[:sv₊p][1]")`.
- a single mean random effect, e.g. `Symbol("random[:sv₊τ]")`.
- an uncertainty parameter of the random effect, e.g. `Symbol("prand_σ[:sv₊τ]")`.
- a individual effect for each individual, e.g. `Symbol("indiv[:sv₊i, 3]")` 
  for the third individual.
- the individual multiplier for the random effect for each individual,
  e.g. `Symbol("indiv_random[:sv₊τ, 3]")`.

## Extracting individual effects 

Each row of a multivariate chain can be extracted as a ComponentVector
as described in [Extracting effects from sampled object].

```@example doc
chn2 = chn[:,vcat(pop_info.effect_pos[:indiv_random][:B]...),:]
chn3 = extract_group(chn2, :indiv_random, indiv_ids)
names(chn3)
```











