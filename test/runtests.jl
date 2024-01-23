using ChainRulesCore
using ChainRulesTestUtils
using DiffPointRasterisation
using Rotations
using StaticArrays
using Test
using TestItemRunner

@testset "ChainRules" begin
    @testset "single" begin
        grid_size = (8, 8, 8)
        points = 0.3 .* randn(3, 10)
        rotation = Array(rand(RotMatrix3))
        translation = 0.1 .* randn(3)
        background = 0.1
        weight = 1.0

        test_rrule(raster, grid_size, points, rotation, translation, background, weight)

        # default arguments
        test_rrule(raster, grid_size, points, rotation, translation)

        grid_size = (8, 8)
        translation = 0.1 .* randn(2)

        test_rrule(raster_project, grid_size, points, rotation, translation, background, weight)

        # default arguments
        test_rrule(raster_project, grid_size, points, rotation, translation)
    end

    @testset "batch" begin
        batch_size = 3
        grid_size = (8, 8, 8)
        points = 0.3 .* randn(3, 10)
        rotation = [Array(rand(RotMatrix3)) for _ in 1:batch_size]
        translation = [0.1 .* randn(3) for _ in 1:batch_size]
        background = fill(0.1, batch_size)
        weight = fill(1.0, batch_size)

        test_rrule(raster, grid_size, points, rotation, translation, background, weight)

        # default arguments
        test_rrule(raster, grid_size, points, rotation, translation)

        grid_size = (8, 8)
        translation = [0.1 .* randn(2) for _ in 1:batch_size]

        test_rrule(raster_project, grid_size, points, rotation, translation, background, weight)

        # default arguments
        test_rrule(raster_project, grid_size, points, rotation, translation)
    end
end

@run_package_tests