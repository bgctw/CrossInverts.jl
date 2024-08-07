```@meta
CurrentModule = CrossInverts
```

# Extracting effects from sampled object

When setting up the inversion using [`setup_inversion`](@ref), the ComponentVector
`pop_info.effect_pos` is returned, which specifies the positions of effects in each sample.

First lets mockup a sampling result and corresponding effect positions.
```@example doc
using CrossInverts
using MCMCChains
using ComponentArrays: ComponentArrays as CA


chn = Chains(randn((10,21,2)), ["fixed[:p][1]", "fixed[:p][2]", "fixed[:p][3]", "random[:x][1]", "random[:x][2]", "random[:τ]", "prand_σ[:x][1]", "prand_σ[:x][2]", "prand_σ[:τ]", "indiv[:i, 1]", "indiv[:i, 2]", "indiv[:i, 3]", "indiv_random[:x, 1][1]", "indiv_random[:x, 1][2]", "indiv_random[:τ, 1]", "indiv_random[:x, 2][1]", "indiv_random[:x, 2][2]", "indiv_random[:τ, 2]", "indiv_random[:x, 3][1]", "indiv_random[:x, 3][2]", "indiv_random[:τ, 3]"])

pop_info = (;
    effect_pos = CA.ComponentVector(fixed = (p = 1:3,), random = (x = 4:5, τ = 6), 
        random_σ = (x = 7:8, τ = 9), indiv = (A = (i = 10), B = (i = 11), 
        C = (i = 12)), indiv_random = (A = (x = 13:14, τ = 15), 
        B = (x = 16:17, τ = 18), C = (x = 19:20, τ = 21))),
    indiv_ids = (:A, :B, :C))
nothing # hide
```

The `effect_pos` can be conveniently used to index into a sample, e.g. to
extract sub-Chains from the sampling results.
Lets get the random multipliers for individual `:B`, the vector
parameter `:x` and scalar `τ`.

```@example doc
pop_info.effect_pos[:indiv_random][:B]
```
```@example doc
chn2 = chn[:,vcat(pop_info.effect_pos[:indiv_random][:B]...),:]
names(chn2)
```

The names can be simplified by constraining to a single group
using [`extract_group`](@ref). In addition, this allows replacing the indices of 
individuals by more readable identifiers.

```@example doc
chn3 = extract_group(chn2, :indiv_random, pop_info.indiv_ids)
names(chn3)
```

Alternatively, one can attach the ComponentArrays-Axis 
to the array constructed from the chain and index into it.

```@example doc
s1 = CA.ComponentMatrix(Array(chn),
    CA.FlatAxis(), first(CA.getaxes(pop_info.effect_pos)))
# sv.p within fixed
s1[:, :fixed][:, :p]
# random effects multiplier for site B for random parameter tau
s1[:, :indiv_random][:, :B][:, :τ]
nothing # hide
```

## API
```@docs
extract_group
```







