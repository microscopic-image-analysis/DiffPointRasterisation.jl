using ChainRulesCore
using ChainRulesTestUtils
using DiffPointRasterisation
using Rotations
using StaticArrays
using Test
using TestItemRunner

include("../src/testing.jl")

const grid_size_3d = (8, 8, 8)
const grid_size_2d = (8, 8)
const batch_size = batch_size_for_test()
const points = 0.5 .* randn(3, 10)
const rotation = Array(rand(RotMatrix3))
const rotation_tangent = Array(rand(RotMatrix3))
const rotations = stack(rand(RotMatrix3, batch_size))
const rotation_tangents = stack(rand(RotMatrix3, batch_size))
const translation_3d = 0.1 .* randn(3)
const translation_2d = 0.1 .* randn(2)
const translations_3d = zeros(3, batch_size)
const translations_2d = zeros(2, batch_size)
const background = 0.1
const backgrounds = collect(1:1.0:batch_size)
const weight = 1.0
const weights = 10 .* ones(batch_size)

@testset "ChainRules" begin
    @testset "single" begin
        test_rrule(raster, grid_size_3d, points, rotation ⊢ rotation_tangent, translation_3d, background, weight)

        # default arguments
        test_rrule(raster, grid_size_3d, points, rotation ⊢ rotation_tangent, translation_3d)


        test_rrule(raster_project, grid_size_2d, points, rotation ⊢ rotation_tangent, translation_2d, background, weight)

        # default arguments
        test_rrule(raster_project, grid_size_2d, points, rotation ⊢ rotation_tangent, translation_2d)
    end

    @testset "batch" begin
        test_rrule(raster, grid_size_3d, points, rotations ⊢ rotation_tangents, translations_3d, backgrounds, weights)

        # default arguments
        test_rrule(raster, grid_size_3d, points, rotations ⊢ rotation_tangents, translations_3d)


        test_rrule(raster_project, grid_size_2d, points, rotations ⊢ rotation_tangents, translations_2d, backgrounds, weights)

        # default arguments
        test_rrule(raster_project, grid_size_2d, points, rotations ⊢ rotation_tangents, translations_2d)
    end
end

    end
end

@run_package_tests