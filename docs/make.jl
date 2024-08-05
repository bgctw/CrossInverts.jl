using CrossInverts
using Documenter

DocMeta.setdocmeta!(CrossInverts, :DocTestSetup, :(using CrossInverts); recursive = true)

makedocs(;
    #modules = [CrossInverts], # uncomment for errors on docstrings not included
    authors = "Thomas Wutzler <twutz@bgc-jena.mpg.de> and contributors",
    repo = Remotes.GitHub("bgctw", "CrossInverts.jl"),
    sitename = "CrossInverts.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://bgctw.github.io/CrossInverts.jl",
        edit_link = "main",
        assets = String[],),
    pages = [
        "Home" => "index.md",
        "Walkthrough" => "example_vec.md", # move to start when not testing docu
        "Extracting effects" => "extract_groups.md",
        "Providing inversion information" => "inversion_case.md",
    ],)

deploydocs(;
    repo = "github.com/bgctw/CrossInverts.jl",
    devbranch = "main",)
