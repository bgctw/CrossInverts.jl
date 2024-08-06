"""
    extract_group(chn, group::Symbol) 
    extract_group(chn, group::Symbol, indiv_ids) 

Extract SubChain of components (`:fixed`, `:random`, `:prand_σ`, `:indiv`, `:indiv_random`)
and replace parameter names like `group[:p1, o1]o2 => group[o1]o2`.
The variant, where `indiv_ids` is specified, additionally 
replaces first `[i]` by `[:(indiv_ids[i])]`.
"""
function extract_group(chn::MCMCChains.Chains, group::Symbol) 
  _names = string.(MCMCChains.namesingroup(chn, group))
  _compstr = string(group)
  _dict = Dict(_names .=> replace.(_names , Regex("$_compstr\\[(.+?)\\]") => s"\1"))
  tmp = @chain chn begin
      MCMCChains.group(group)
      MCMCChains.replacenames(_dict)
  end
  _names2 = string.(names(tmp))
  _dict2 = Dict(_names2 .=> replace.(_names2 , r",\s*(\d+)?" .=> s"[\1]"))
  MCMCChains.replacenames(tmp, _dict2)
end

function extract_group(chn::MCMCChains.Chains, group::Symbol, indiv_ids) 
  tmp = extract_group(chn, group)
  replace_indiv_ids(tmp, indiv_ids)
end

function replace_indiv_ids(chn::MCMCChains.Chains, indiv_ids)
  _names = string.(names(chn))
  _pat = Regex.("\\[".*string.(1:length(indiv_ids)).*"\\]") .=> "[:" .* string.(indiv_ids) .* "]"
  _dict = Dict(_names .=> replace.(_names, _pat...; count=1))
  MCMCChains.replacenames(chn, _dict)
end


"""
  compute_indiv_random(chn::MCMCChains.Chains, indiv_ids) 

Extract random and random_indiv Subchains and compute the restuling parameter. 
It multiplies the mean random effect by the individual multiplicative random effect.
"""
function compute_indiv_random(chn::MCMCChains.Chains, indiv_ids) 
  _means = extract_group(chn, :random)
  _rm = extract_group(chn, Val(:indiv_random), indiv_ids)
  _means_arr = cat(Array(_means, append_chains=false)..., dims=3)
  _rm_arr = cat(Array(_rm, append_chains=false)..., dims=3)
  _means_arr_filled = hcat(fill(_means_arr, length(indiv_ids))...)
  MCMCChains.Chains(_means_arr_filled .* _rm_arr, names(_rm))
end

