using Test
using CrossInverts
using CrossInverts: CrossInverts as CP
using MCMCChains

moduledir = dirname(dirname(pathof(CrossInverts)))
testdir = joinpath(moduledir, "test")
include(joinpath(testdir,"example_chns.jl"))

#First lets mockup a sampling result and correspond effect positions
(; chn, effect_pos) = get_example_chain()
indiv_ids = keys(effect_pos.indiv)

@testset "extract_group" begin
    chn2 = chn[:, vcat(effect_pos[:indiv_ranmul][:B]...), :]
    @test names(chn2) == [Symbol("indiv_ranmul[:sv₊x, 2][1]"),
        Symbol("indiv_ranmul[:sv₊x, 2][2]"), Symbol("indiv_ranmul[:sv₊τ, 2]")]
    #
    chn3 = extract_group(chn, :indiv_ranmul)
    # show(string.(names(chn3)))
    @test names(chn3) ==
          Symbol.([":sv₊x[1][1]", ":sv₊x[1][2]", ":sv₊τ[1]", ":sv₊x[2][1]",
        ":sv₊x[2][2]", ":sv₊τ[2]", ":sv₊x[3][1]", ":sv₊x[3][2]", ":sv₊τ[3]"])
    #
    chn4 = extract_group(chn, :indiv_ranmul, indiv_ids)
    # show(string.(names(chn4)))
    @test names(chn4) ==
          Symbol.([":sv₊x[:A][1]", ":sv₊x[:A][2]", ":sv₊τ[:A]", ":sv₊x[:B][1]",
        ":sv₊x[:B][2]", ":sv₊τ[:B]", ":sv₊x[:C][1]", ":sv₊x[:C][2]", ":sv₊τ[:C]"])
end;
