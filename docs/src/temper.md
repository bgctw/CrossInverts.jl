```@meta
CurrentModule = CrossInverts
```

# Using tempered/inflated uncertainty to account for model uncertainty

The usual cost function based on Normal distribution is a sum 
across $({y_{obs}}_i - {y_{pred}}_i)^2/{σ_r^2}_i$ where usually
 one assumes tha residual uncertainty equals variance of
observation uncertainty: ${σ_r^2}_i = {σ_o^2}_i$.
This assumes that there is a perfect model and uncertainty is only due to
observation noise.
However, a part of model-data residuals is due to model uncertainty, ${σ_o^2}_i$.

We approximate this increase of residual variance at stream level by 
multiplying the average observation uncertainty by a temperature, $T ≥ 1$:
${σ^2_r}_a = {σ^2_o}_a + {σ^2_m}_a = T {σ^2_o}_a$.

CrossInverts uses ${σ^2_r}_i = T {σ_o^2}_i$ computes temperature, 
$$
T_s = 1 + n_s w_s r
$$
, for stream, $s$, given 
- number of observations, $n$,
- weight of allocating model uncertainty aross streams, $w_s$, and
- assumed model uncertainty as as factor, $r$ of observation uncertainty.

We recommend using a single weight for a given observation stream
across replicates.

The following section derives this relationship, but can be 
skipped for application.
 
## Derivation of stream temperature

Because the observation uncertainty declines with number of observations, 
$n$, but model uncertainty does not we write
${σ^2_r}/n = {σ^2_o}/n + {σ^2_m} = T {σ^2_o}/n$.

With expressing model uncertainty as a factor of observation uncertainty we get.
- $r = {σ^2_m} / {σ^2_o}$
- ${σ^2_r}/n = {σ^2_o}/n + {σ^2_m} = T {σ^2_o}/n$
- $1 + n r = T$

Or with a weighting model uncertainy 
- $w_i r = {σ^2_m} / {σ^2_o}$
- $1 + n \, w_i \, r = T$

## Estimating model uncertainty fraction, $r$

Given a model prediction, the variance of the residuals
can be compared to observations to get an estimate of r for each 
stream.

$$
r = \frac{1}{n w_s} \left( \frac{{σ^2_r}}{{σ^2_o}} -1 \right)
$$

derived from
$$
{σ^2_m} = {σ^2_r}_a - {σ^2_o}_a = w_s r {σ^2_o} \\
{σ^2_r}/n - {σ^2_o}/n = w_s r  {σ^2_o}\\
\frac{1}{n w_s} \frac{{σ^2_r} - {σ^2_o}}{{σ^2_o}} = r\\
$$

## Estimating stream weights, $w_i$

One possible notion of proportion of allowed relative model error,
is that is should be a similar fraction of the range of data.


