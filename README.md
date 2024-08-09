# CrossInverts

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bgctw.github.io/CrossInverts.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bgctw.github.io/CrossInverts.jl/dev/)
[![Build Status](https://github.com/bgctw/CrossInverts.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bgctw/CrossInverts.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/bgctw/CrossInverts.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/bgctw/CrossInverts.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


## Problem

When calibrating models across different individuals, 
some parameters should be kept the same constant in all individuals (fixed parameters), 
some are allowed to vary around a common mean across individuals (random parameters), 
and others may be allowed to vary freely (individual parameters).

This package helps with specifying priors and and dealing with population and individuals Further, it helps providing different specification by different scenarios.

See the [walkthrough](https://bgctw.github.io/CrossInverts.jl/dev/example_vec/)
in the documentation for example code.

## Inversion specification
The user needs to specify for a given scenario
- the model, i.e. ODESystem to be inverted at each individual
- which parameters are optimized as fixed, random, and individual
- which individuals take part in the inversion
- priors of parameters and priors of uncertainty of random effects
- observations and associated uncertainty

and optionally
- initial states and parameters (or let them infer from priors)
- a `ProblemUpdater` in case changes of optimized parameters propagate to other
  parameters

## Model creation
Based those specification the package creates
- a function for forward simulation given a realization of fixed and random effects.
- a Turing model to sample the posterior of the effects

## Extracting results
The package helps to access specific effects and quantities for specific individuals 
in the list of sampled 
