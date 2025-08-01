using Test, SafeTestsets
const GROUP = get(ENV, "GROUP", "All") # defined in in CI.yml
@show GROUP

@time begin
    if GROUP == "All" || GROUP == "Basic"
        #@safetestset "Tests" include("test/test_samplesystem_vec.jl")
        @time @safetestset "test_samplesystem_vec" include("test_samplesystem_vec.jl")
        # #@safetestset "Tests" include("test/test_indivdata.jl")
        # @time @safetestset "test_indivdata" include("test_indivdata.jl")
        #@safetestset "Tests" include("test/test_indivdata_vec.jl")
        @time @safetestset "test_indivdata_vec" include("test_indivdata_vec.jl")
        #@safetestset "Tests" include("test/test_util_mixed.jl")
        @time @safetestset "test_util_mixed" include("test_util_mixed.jl")
        #@safetestset "Tests" include("test/test_extract_group.jl")
        @time @safetestset "test_extract_group" include("test_extract_group.jl")
        #@safetestset "Tests" include("test/test_ranadd.jl")
        @time @safetestset "test_ranadd" include("test_ranadd.jl")
    end
    if GROUP == "All" || GROUP == "JET"
        #@safetestset "Tests" include("test/test_JET.jl")
        @time @safetestset "test_JET" include("test_JET.jl")
        #@safetestset "Tests" include("test/test_aqua.jl")
        @time @safetestset "test_Aqua" include("test_aqua.jl")
    end
end
