
@testitem "ChainRules single" begin
    using ChainRulesTestUtils, ChainRulesCore
    include("data.jl")

    test_rrule(raster, D.grid_size_3d, D.points_static, D.rotation ⊢ D.rotation_tangent, D.translation_3d, D.background, D.weight)

    # default arguments
    test_rrule(raster, D.grid_size_3d, D.points_static, D.rotation ⊢ D.rotation_tangent, D.translation_3d)


    test_rrule(raster, D.grid_size_2d, D.points_static, D.projection ⊢ D.projection_tangent, D.translation_2d, D.background, D.weight)

    # default arguments
    test_rrule(raster, D.grid_size_2d, D.points_static, D.projection ⊢ D.projection_tangent, D.translation_2d)
end

@testitem "ChainRules batch" begin
    using ChainRulesTestUtils
    include("data.jl")

    test_rrule(raster, D.grid_size_3d, D.points_static, D.rotations_static ⊢ D.rotation_tangents_static, D.translations_3d_static, D.backgrounds, D.weights)

    # default arguments
    test_rrule(raster, D.grid_size_3d, D.points_static, D.rotations_static ⊢ D.rotation_tangents_static, D.translations_3d_static)


    test_rrule(raster, D.grid_size_2d, D.points_static, D.projections_static ⊢ D.projection_tangents_static, D.translations_2d_static, D.backgrounds, D.weights)

    # default arguments
    test_rrule(raster, D.grid_size_2d, D.points_static, D.projections_static ⊢ D.projection_tangents_static, D.translations_2d_static)
end