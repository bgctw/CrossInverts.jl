"""
    extract_group(chn, group::Symbol) 
    extract_group(chn, group::Symbol, indiv_ids) 

Extract SubChain of components 
(`:fixed`, `:ranmul`, `:ranadd`, `:pranmul_σ`, `:pranadd_σ`, `:indiv`, `:indiv_ranmul`, `:indiv_ranadd`)
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

Extract `ranmul` and `ranmul_indiv` subchains and compute the resulting parameter. 
It multiplies the mean ranmul effect by the individual multiplicative ranmul effect.

Further, extract `ranadd` and `ranadd_indiv` subchains and compute the resulting parameter. 
It add the mean ranadd and the individual additive ranadd effect.

"""
function compute_indiv_random(chn::MCMCChains.Chains, indiv_ids) 
  _means = extract_group(chn, :ranmul)
  _rm = extract_group(chn, Val(:indiv_ranmul), indiv_ids)
  _means_arr = cat(Array(_means, append_chains=false)..., dims=3)
  _rm_arr = cat(Array(_rm, append_chains=false)..., dims=3)
  _means_arr_filled = hcat(fill(_means_arr, length(indiv_ids))...)
  _arr_mul = _means_arr_filled .* _rm_arr
  #
  _means = extract_group(chn, :ranadd)
  _ra = extract_group(chn, Val(:indiv_ranadd), indiv_ids)
  _means_arr = cat(Array(_means, append_chains=false)..., dims=3)
  _ra_arr = cat(Array(_ra, append_chains=false)..., dims=3)
  _means_arr_filled = hcat(fill(_means_arr, length(indiv_ids))...)
  _arr_add = _means_arr_filled .+ _ra_arr
  #
  MCMCChains.Chains(hcat(_arr_add, _arr_mul), vcat(names(_ra), names(_rm)))
end

