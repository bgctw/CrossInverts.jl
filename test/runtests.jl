tmpf = () -> begin
    push!(LOAD_PATH, expanduser("~/julia/devtools/")) # access local pack
    push!(LOAD_PATH, expanduser("~/julia/scimltools/")) # access local pack
    push!(LOAD_PATH, joinpath(pwd(), "test/")) # access local pack
end

using Test, SafeTestsets
const GROUP = get(ENV, "GROUP", "All") # defined in in CI.yml
@show GROUP

@time begin
    if GROUP == "All" || GROUP == "Basic"
    end
    if GROUP == "All" || GROUP == "JET"
        #@safetestset "Tests" include("test/test_JET.jl")
        @time @safetestset "test_JET" include("test_JET.jl")
        #@safetestset "Tests" include("test/test_aqua.jl")
        @time @safetestset "test_Aqua" include("test_aqua.jl")
    end
end
