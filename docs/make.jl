using CrossInverts
using Documenter

DocMeta.setdocmeta!(CrossInverts, :DocTestSetup, :(using CrossInverts); recursive=true)

makedocs(;
    modules=[CrossInverts],
    authors="Thomas Wutzler <twutz@bgc-jena.mpg.de> and contributors",
    repo="https://github.com/bgctw/CrossInverts.jl/blob/{commit}{path}#{line}",
    sitename="CrossInverts.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bgctw.github.io/CrossInverts.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bgctw/CrossInverts.jl",
    devbranch="main",
)
