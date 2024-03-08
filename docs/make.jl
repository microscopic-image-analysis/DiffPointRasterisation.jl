using Documenter
using DiffPointRasterisation

makedocs(;
    sitename="DiffPointRasterisation",
    format=Documenter.HTML(),
    modules=[DiffPointRasterisation],
    pages=[
        "Home" => "index.md",
        "Batch of poses" => "batch.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/trahflow/DiffPointRasterisation.jl.git",
    devbranch="main",
    push_preview=true,
)