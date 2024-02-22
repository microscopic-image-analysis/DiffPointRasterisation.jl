
@testitem "CUDA forward" begin
    using Adapt, CUDA
    CUDA.allowscalar(false)
    include("data.jl")
    include("util.jl")
    cuda_available = CUDA.functional()
    
    args = (D.grid_size_3d, D.points, D.rotations, D.translations_3d, D.backgrounds, D.weights)
    @test cuda_cpu_agree(raster, args...) skip=!cuda_available

    args = (D.grid_size_2d, D.points, D.rotations, D.translations_2d, D.backgrounds, D.weights)
    @test cuda_cpu_agree(raster_project, args...) skip=!cuda_available
end

@testitem "CUDA backward" begin
    using Adapt, CUDA
    CUDA.allowscalar(false)
    include("data.jl")
    include("util.jl")
    cuda_available = CUDA.functional()

    ds_dout_3d = randn(D.grid_size_3d..., D.batch_size)
    args = (ds_dout_3d, D.points, D.rotations, D.translations_3d, D.backgrounds, D.weights)
    @test cuda_cpu_agree(raster_pullback!, args...) skip=!cuda_available

    ds_dout_2d = randn(D.grid_size_2d..., D.batch_size)
    args = (ds_dout_2d, D.points, D.rotations, D.translations_2d, D.backgrounds, D.weights)
    @test cuda_cpu_agree(raster_project_pullback!, args...) skip=!cuda_available
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
#         test_rrule(raster_project, args...; output_tangent=ds_dout_2d)
#     end
# end
