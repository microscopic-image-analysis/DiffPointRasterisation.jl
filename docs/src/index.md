# DiffPointRasterisation

*Differentiable rasterisation of point clouds in julia*

DiffPointRasterisation.jl provides a rasterisation routine for arbitrary-dimensional point cloud data that is fully (auto-)differentiable.
The implementation uses multiple threads on CPU or GPU hardware if available.

## Rasterisation interface

The interface consists of a single function [`raster`](@ref) that accepts a point cloud (as a vector of m-dimensional vectors) and pose/projection parameters, (as well as optional weight and background parameters) and returns a n-dimensional (n <= m) array into which the points are rasterized, each point by default with a weight of 1 that is mulit-linearly interpolated into the neighboring grid cells.

## Differentiability

Both, an explicit function that calculates derivatives of `raster`, as well as an integration to common automatic differentiation libraries in julia are provided.

### Automatic differentiation libraries

Rules for reverse-mode automatic differentiation libraries that are based on the [ChainRules.jl](https://juliadiff.org/ChainRulesCore.jl/dev/#ChainRules-roll-out-status) ecosystem are provided via an extension package. So using `raster(args...)` in a program that uses any of the ChainRules-based reverse-mode autodiff libraries should just work™. Gradients with respect to all parameters (except `grid_size`) are supported.

### Explicit interface

The explicit interface for calculating derivatives of `raster` with respect to its arguments again consists of a single function called [`raster_pullback!`](@ref):

The function `raster_pullback!(ds_dout, raster_args...)` takes as input the sensitivity of some scalar quantity to the output of `raster(grid_size, raster_args...)`, `ds_dout`, and returns the sensitivity of said quantity to the *input arguments* `raster_args` of `raster` (hence the name pullback).
