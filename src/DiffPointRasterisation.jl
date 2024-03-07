module DiffPointRasterisation

using ArgCheck
using Atomix
using ChunkSplitters
using FillArrays
using KernelAbstractions
using SimpleUnPack
using StaticArrays
using TestItems

include("util.jl")
include("raster.jl")
include("raster_pullback.jl")
include("interface.jl")

export raster, raster!, raster_pullback!

end
