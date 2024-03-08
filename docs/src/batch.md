# Raster a single point cloud to a batch of poses

To make best use of the hardware it is advantageous to raster a batch of poses at once.
On GPU hardware this is currently also the only supported mode.

To raster a single point cloud to a batch of `n` images, all parameters except the point cloud should be provided as `n`-vectors.

This is a more flexible interface than the often used array with trailing batch dimension, since it allows to pass in a batch of parameters that have a more structured type than a simple array (e.g. a vector of `Rotation` objects from [Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl)).

## Array with trailing batch dim to vec of array

If you have data in the array with trailing batch dimension format, it is straightforward (and quite cheap) to reinterpret it as a batch-vector of single parameters:

```julia-repl
julia> matrices = randn(2, 2, 3)  # batch of 3 2x2-matrices as 3d-array
2×2×3 Array{Float64, 3}:
[:, :, 1] =
 -0.947072  1.10155
  0.328925  0.0957267

[:, :, 2] =
 -1.14336   1.71218
  0.277723  0.436665

[:, :, 3] =
 -0.114541  -0.769275
  0.321084  -0.215008

julia> using StaticArrays

julia> vec_of_matrices = reinterpret(reshape, SMatrix{2, 2, Float64, 4}, reshape(matrices, 4, :))
3-element reinterpret(reshape, SMatrix{2, 2, Float64, 4}, ::Matrix{Float64}) with eltype SMatrix{2, 2, Float64, 4}:
 [-0.947072487060636 1.1015531033643386; 0.3289251820481776 0.0957267306067441]
 [-1.143363316882325 1.712179045069409; 0.27772320359678004 0.4366650562384542]
 [-0.11454148373779363 -0.7692750798350269; 0.32108447348937047 -0.21500805160408776]
```

## Pre-allocation for batched pullback

[`raster_pullback!`](@ref) can be optionally provided with pre-allocated arrays for its output.
For these arrays the expected format is actually in the nd-array with trailing batch dimension format.
The rationale behind this is that the algorithm works better on continuous blocks of memory, since atomic operations are required. 
