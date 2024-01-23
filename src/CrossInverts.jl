module CrossInverts

using OrdinaryDiffEq, ModelingToolkit
using MTKHelpers
using DataFrames
using Tables: Tables
using ComponentArrays
using Distributions, DistributionFits
using Chain
using LoggingExtras
using Test: @inferred
using StaticArrays
using Turing: Turing
using MCMCChains: MCMCChains
using Random, StableRNGs
using PDMats: PDiagMat

export AbstractCrossInversionCase
export setup_tools_indiv, get_indivdata, get_priors_dict, dict_to_cv
export get_indiv_parameters_from_priors
include("site_data.jl")

export SampleSystem1Case
include("example_system1.jl")

export SampleSystemVecCase
include("example_system_vec.jl")

export setup_psets_fixed_random_indiv, gen_sim_sols_probs, gen_sim_sols
export setup_priors_pop
# export gen_compute_indiv_rand
export get_obs_uncertainty_dist_type
include("util_mixed.jl")

export setup_tools_mixed
export gen_model_cross
include("util_sample.jl")

end
