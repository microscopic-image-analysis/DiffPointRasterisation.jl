# DiffPointRasterisation

*Differentiable rasterisation of point clouds in julia*

[![Build Status](https://github.com/trahflow/DiffPointRasterisation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/trahflow/DiffPointRasterisation.jl/actions/workflows/CI.yml?query=branch%3Amain)
 [![](https://img.shields.io/badge/docs-main-blue.svg)](https://trahflow.github.io/DiffPointRasterisation.jl/dev)

![](logo.gif)

## About

This package provides a rasterisation routines for arbitrary-dimensional point cloud data that is fully (auto-)differentiable.
The implementation uses multiple threads on CPU or GPU hardware if available.

## Rasterisation interface

The interface consists of a single function `raster` that accepts a point cloud (as a vector-of-vectors) and pose/projection parameters, (as well as optional weight and background parameters) and returns a n-dimensional array into which the points are rasterized, each point by default with a weight of 1 that is mulit-linearly interpolated into the neighboring grid cells.

#### Examples

```julia-repl
julia> using DiffPointRasterisation, LinearAlgebra

julia> grid_size = (5, 5)  # 2d grid with 5 x 5 pixels
(5, 5)

julia> rotation, translation = I(2), zeros(2)  # pose parameters
(Bool[1 0; 0 1], [0.0, 0.0])
```

```julia-repl
julia> raster(grid_size, [zeros(2)], rotation, translation)  # single point at center
5×5 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```

```julia-repl
julia> raster(grid_size, [[0.2, 0.0]], rotation, translation)  # single point half a pixel below center
5×5 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.5  0.0  0.0
 0.0  0.0  0.5  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```

```julia-repl
julia> raster(grid_size, [[0.2, -0.2]], I(2), zeros(2))  # single point half a pixel below and left of center
5×5 Matrix{Float64}:
 0.0  0.0   0.0   0.0  0.0
 0.0  0.0   0.0   0.0  0.0
 0.0  0.25  0.25  0.0  0.0
 0.0  0.25  0.25  0.0  0.0
 0.0  0.0   0.0   0.0  0.0
```

## Differentiability

### Explicit interface

### Automatic differentiation libraries

This package provides rules for reverse-mode automatic differentiation libraries that are based on the [ChainRules.jl](https://juliadiff.org/ChainRulesCore.jl/dev/#ChainRules-roll-out-status) ecosystem. So using `raster(args...)` in a program that uses any of the ChainRules-based reverse-mode autodiff libraries should just work™. Gradients with respect to all parameters (except of course `grid_size`) are supported:

#### Example using Zygote.jl

```julia-repl
julia> using Zygote

julia> target_image = rand(grid_size...)
5×5 Matrix{Float64}:
 0.0190286  0.404414  0.54493     0.988256  0.191955
 0.819837   0.426972  0.00210893  0.78712   0.98477
 0.729581   0.281671  0.678572    0.424984  0.130405
 0.904558   0.866291  0.612202    0.73597   0.473529
 0.112324   0.232005  0.883618    0.806333  0.190192

julia> loss(params...) = sum((target_image .- raster(grid_size, params...)).^2)
loss (generic function with 1 method)

julia> points = [2 * rand(2) .- 1 for _ in 1:5]  # 5 random points
5-element Vector{Vector{Float64}}:
 [0.8457397177007744, 0.3482756109584688]
 [-0.6028188536164718, -0.612801322279686]
 [-0.47141692007256464, 0.6098964840013308]
 [-0.74526926786903, 0.6480225109030409]
 [-0.4044384373422192, -0.13171854413805173]

julia> rotation = [  # explicit matrix for rotation (to satisfy Zygote)
           1.0 0.0
           0.0 1.0
       ]
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0

julia> d_points, d_rotation, d_translation = Zygote.gradient(loss, points, rotation, translation);

julia> d_points
5-element Vector{StaticArraysCore.SVector{2, Float64}}:
 [0.6505328781226858, 3.249835880979481]
 [-1.4648814811123378, 0.6431239277807955]
 [-1.8283719289689442, 0.39386232318301656]
 [-2.3700204282508586, 4.284425417816211]
 [4.501808783874736, 2.6086546339580963]

julia> d_rotation
2×2 reshape(::StaticArraysCore.SMatrix{2, 2, Float64, 4}, 2, 2) with eltype Float64:
  2.24076  -2.11967
 -2.07294   3.41074

julia> d_translation
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 -0.5109321763347179
 11.1799021837176
```