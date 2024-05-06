
@testitem "CUDA forward" begin
    using Adapt, CUDA
    CUDA.allowscalar(false)
    include("data.jl")
    include("util.jl")
    cuda_available = CUDA.functional()

    # no projection
    args = (
        D.grid_size_3d,
        D.more_points,
        D.rotations_static,
        D.translations_3d_static,
        D.backgrounds,
        D.weights,
        D.more_point_weights,
    )
    @test cuda_cpu_agree(raster, args...) skip = !cuda_available

    # default arguments
    args = (D.grid_size_3d, D.more_points, D.rotations_static, D.translations_3d_static)
    @test cuda_cpu_agree(raster, args...) skip = !cuda_available

    # projection
    args = (
        D.grid_size_2d,
        D.more_points,
        D.projections_static,
        D.translations_2d_static,
        D.backgrounds,
        D.weights,
        D.more_point_weights,
    )
    @test cuda_cpu_agree(raster, args...) skip = !cuda_available
end

@testitem "CUDA backward" begin
    using Adapt, CUDA
    CUDA.allowscalar(false)
    include("data.jl")
    include("util.jl")
    cuda_available = CUDA.functional()

    # no projection
    ds_dout_3d = randn(D.grid_size_3d..., D.batch_size)
    args = (
        ds_dout_3d,
        D.more_points,
        D.rotations_static,
        D.translations_3d_static,
        D.backgrounds,
        D.weights,
        D.more_point_weights,
    )
    @test cuda_cpu_agree(raster_pullback!, args...) skip = !cuda_available

    # default arguments
    args = (ds_dout_3d, D.more_points, D.rotations_static, D.translations_3d_static)
    @test cuda_cpu_agree(raster_pullback!, args...) skip = !cuda_available

    # projection
    ds_dout_2d = randn(D.grid_size_2d..., D.batch_size)
    args = (
        ds_dout_2d,
        D.more_points,
        D.projections_static,
        D.translations_2d_static,
        D.backgrounds,
        D.weights,
        D.more_point_weights,
    )
    @test cuda_cpu_agree(raster_pullback!, args...) skip = !cuda_available
end

# The follwing currently fails.
# Not sure whether test_rrule is supposed to play nicely with CUDA.

# @testitem "CUDA ChainRules" begin
#     using Adapt, CUDA, ChainRulesTestUtils
#     include("data.jl")
#     include("util.jl")
#     c(a) = adapt(CuArray, a)
#     if CUDA.functional()
#         ds_dout_3d = CUDA.randn(Float64, D.grid_size_3d..., D.batch_size)
#         args = (D.grid_size_3d, c(D.points), c(D.rotations) ⊢ c(D.rotation_tangents), c(D.translations_3d), c(D.backgrounds), c(D.weights))
#         test_rrule(raster, args...; output_tangent=ds_dout_3d)
# 
#         ds_dout_2d = CUDA.randn(Float64, D.grid_size_2d..., D.batch_size)
#         args = (D.grid_size_2d, c(D.points), c(D.rotations) ⊢ c(D.rotation_tangents), c(D.translations_2d), c(D.backgrounds), c(D.weights))
#         test_rrule(raster, args...; output_tangent=ds_dout_2d)
#     end
# end
