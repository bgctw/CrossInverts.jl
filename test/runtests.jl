tmpf = () -> begin
    pop!(LOAD_PATH)
    push!(LOAD_PATH, joinpath(pwd(), "test/"))
    push!(LOAD_PATH, expanduser("~/julia/devtools_$(VERSION.major).$(VERSION.minor)"))
end

using Test, SafeTestsets
const GROUP = get(ENV, "GROUP", "All") # defined in in CI.yml
@show GROUP

@time begin
    if GROUP == "All" || GROUP == "Basic"
        #@safetestset "Tests" include("test/test_samplesystem_vec.jl")
        @time @safetestset "test_samplesystem_vec" include("test_samplesystem_vec.jl")
        #@safetestset "Tests" include("test/test_sitedata.jl")
        @time @safetestset "test_sitedata" include("test_sitedata.jl")
        #@safetestset "Tests" include("test/test_sitedata_vec.jl")
        @time @safetestset "test_sitedata_vec" include("test_sitedata_vec.jl")
    end
    if GROUP == "All" || GROUP == "JET"
        #@safetestset "Tests" include("test/test_JET.jl")
        @time @safetestset "test_JET" include("test_JET.jl")
        #@safetestset "Tests" include("test/test_aqua.jl")
        @time @safetestset "test_Aqua" include("test_aqua.jl")
    end
end
