
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
