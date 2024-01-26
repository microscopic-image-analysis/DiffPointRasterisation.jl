using ChainRulesCore
using ChainRulesTestUtils
using DiffPointRasterisation
using Rotations
using StaticArrays
using Test
using TestItemRunner

include("../src/testing.jl")

@testset "ChainRules" begin
    @testset "single" begin
        grid_size = (8, 8, 8)
        points = 0.3 .* randn(3, 4)
        rotation = Array(rand(RotMatrix3))
        rotation_tangent = Array(rand(RotMatrix3))
        translation = 0.1 .* randn(3)
        background = 0.1
        weight = 1.0

        test_rrule(raster, grid_size, points, rotation ⊢ rotation_tangent, translation, background, weight)

        # default arguments
        test_rrule(raster, grid_size, points, rotation ⊢ rotation_tangent, translation)

        grid_size = (8, 8)
        translation = 0.1 .* randn(2)

        test_rrule(raster_project, grid_size, points, rotation ⊢ rotation_tangent, translation, background, weight)

        # default arguments
        test_rrule(raster_project, grid_size, points, rotation ⊢ rotation_tangent, translation)
    end

    @testset "batch" begin
        batch_size = 2
        grid_size = (8, 8, 8)
        points = 0.3 .* randn(3, 4)
        rotation = stack(rand(RotMatrix3, batch_size))
        rotation_tangent = stack(rand(RotMatrix3, batch_size))
        translation = 0.1 .* randn(3, batch_size)
        background = fill(0.1, batch_size)
        weight = fill(1.0, batch_size)

        test_rrule(raster, grid_size, points, rotation ⊢ rotation_tangent, translation, background, weight)

        # default arguments
        test_rrule(raster, grid_size, points, rotation ⊢ rotation_tangent, translation)

        grid_size = (8, 8)
        translation = 0.1 .* randn(2, batch_size)

        test_rrule(raster_project, grid_size, points, rotation ⊢ rotation_tangent, translation, background, weight)

        # default arguments
        test_rrule(raster_project, grid_size, points, rotation ⊢ rotation_tangent, translation)
    end
end

@run_package_tests