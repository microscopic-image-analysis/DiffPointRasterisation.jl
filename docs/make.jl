using Documenter
using DiffPointRasterisation

makedocs(;
    sitename="DiffPointRasterisation",
    format=Documenter.HTML(),
    modules=[DiffPointRasterisation],
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
)

# deploydocs(;
#     repo="github.com/trahflow/DiffPointRasterisation.jl.git",
#     devbranch="main",
#     push_preview=true,
# )