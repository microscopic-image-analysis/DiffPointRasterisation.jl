
@testitem "ChainRules single" begin
    using ChainRulesTestUtils
    include("data.jl")

    test_rrule(raster, D.grid_size_3d, D.points, D.rotation ⊢ D.rotation_tangent, D.translation_3d, D.background, D.weight)

    # default arguments
    test_rrule(raster, D.grid_size_3d, D.points, D.rotation ⊢ D.rotation_tangent, D.translation_3d)


    test_rrule(raster_project, D.grid_size_2d, D.points, D.rotation ⊢ D.rotation_tangent, D.translation_2d, D.background, D.weight)

    # default arguments
    test_rrule(raster_project, D.grid_size_2d, D.points, D.rotation ⊢ D.rotation_tangent, D.translation_2d)
end

@testitem "ChainRules batch" begin
    using ChainRulesTestUtils
    include("data.jl")

    test_rrule(raster, D.grid_size_3d, D.points, D.rotations ⊢ D.rotation_tangents, D.translations_3d, D.backgrounds, D.weights)

    # default arguments
    test_rrule(raster, D.grid_size_3d, D.points, D.rotations ⊢ D.rotation_tangents, D.translations_3d)


    test_rrule(raster_project, D.grid_size_2d, D.points, D.rotations ⊢ D.rotation_tangents, D.translations_2d, D.backgrounds, D.weights)

    # default arguments
    test_rrule(raster_project, D.grid_size_2d, D.points, D.rotations ⊢ D.rotation_tangents, D.translations_2d)
end