###############################################
# Step 6: Actual implementation
###############################################

function raster!(
    out::AbstractArray{T, N_out_p1},
    points::AbstractVector{<:StaticVector{N_in, <:Number}},
    rotation::AbstractVector{<:StaticMatrix{N_out, N_in, <:Number}},
    translation::AbstractVector{<:StaticVector{N_out, <:Number}},
    background::AbstractVector{<:Number},
    weight::AbstractVector{<:Number},
) where {T<:Number, N_in, N_out, N_out_p1}
    @argcheck N_out == N_out_p1 - 1 DimensionMismatch
    out_batch_dim = ndims(out)
    batch_size = size(out, out_batch_dim)
    @argcheck batch_size == length(rotation) == length(translation) == length(background) == length(weight) DimensionMismatch
    n_points = length(points)

    scale = SVector{N_out, T}(size(out)[1:end-1]) / T(2)
    projection_idxs = SVector{N_out}(ntuple(identity, N_out))
    shifts=voxel_shifts(Val(N_out))
    out .= reshape(background, ntuple(_ -> 1, Val(N_out))..., length(background))
    args = (out, points, rotation, translation, weight, shifts, scale, projection_idxs)
    backend = get_backend(out)
    ndrange = (2^N_out, n_points, batch_size)
    workgroup_size = 1024 
    raster_kernel!(backend, workgroup_size, ndrange)(args...)
    out
end

@kernel function raster_kernel!(out::AbstractArray{T}, points, rotations, translations::AbstractVector{<:StaticVector{N_out}}, weights, shifts, scale, projection_idxs) where {T, N_out}
    neighbor_voxel_id, point_idx, batch_idx = @index(Global, NTuple)

    point = @inbounds points[point_idx]
    rotation = @inbounds rotations[batch_idx]
    translation = @inbounds translations[batch_idx]
    weight = @inbounds weights[batch_idx]
    shift = @inbounds shifts[neighbor_voxel_id]
    origin = (-@SVector ones(T, N_out)) - translation

    coord_reference_voxel, deltas = reference_coordinate_and_deltas(
        point,
        rotation,
        projection_idxs,
        origin,
        scale,
    )
    voxel_idx = CartesianIndex(CartesianIndex(Tuple(coord_reference_voxel)) + CartesianIndex(shift), batch_idx)

    if voxel_idx in CartesianIndices(out)
        val = voxel_weight(deltas, shift, projection_idxs, weight)
        @inbounds Atomix.@atomic out[voxel_idx] += val
    end
    nothing
end

"""
    reference_coordinate_and_deltas(point, rotation, projection_idxs, origin, scale)
    
Return 
 - The cartesian coordinate of the voxel of an N-dimensional rectangular 
   grid that is the one closest to the origin, out of the 2^N voxels that are next
   neighbours of the (N-dimensional) `point`
 - A Nx2 array containing coordinate-wise distances of the `scale`d `point` to the
   voxel that is
   * closest to the origin (out of the 2^N next neighbors) in the first column
   * furthest from the origin (out of the 2^N next neighbors) in the second column.

The grid is implicitely assumed to discretize the hypercube ranging from (-1, 1)
in each dimension.
Before `point` is discretized into this grid, it is first translated by 
`-origin` and then scaled by `scale`.
"""
@inline function reference_coordinate_and_deltas(
    point::AbstractVector{T},
    rotation,
    projection_idxs,
    origin,
    scale,
) where {T}
    rotated_point = rotation * point
    projected_point = @inbounds rotated_point[projection_idxs]
    # coordinate of transformed point in output coordinate system
    # which is defined by the (integer) coordinates of the pixels/voxels
    # in the output array.
    coord = (projected_point - origin) .* scale
    # round to **lower** integer (note the -1/2) coordinate ("upper left" if this were a matrix)
    coord_reference_voxel = round.(Int, coord .- T(0.5), RoundUp)
    # distance to lower integer coordinate (distance from "lower left" neighboring pixel
    # in units of fractional pixels):
    deltas_lower = coord - (coord_reference_voxel .- T(0.5))
    # distances to lower (first column) and upper (second column) integer coordinates
    deltas = [deltas_lower one(T) .- deltas_lower]
    coord_reference_voxel, deltas
end

@inline function voxel_weight(deltas, shift, projection_idxs, point_weight)
    lower_upper = mod1.(shift, 2)
    delta_idxs = CartesianIndex.(projection_idxs, lower_upper)
    val = prod(@inbounds @view deltas[delta_idxs]) * point_weight
    val
end

@testitem "raster correctness" begin
    using Rotations
    grid_size = (5, 5)

    points_single_center = [zeros(2)]
    points_single_1pix_right = [[0.0, 0.4]]
    points_single_1pix_up = [[-0.4, 0.0]]
    points_single_1pix_left = [[0.0, -0.4]]
    points_single_1pix_down = [[0.4, 0.0]]
    points_single_halfpix_down = [[0.2, 0.0]]
    points_single_halfpix_down_and_right = [[0.2, 0.2]]
    points_four_cross = reduce(
        vcat,
        [
            points_single_1pix_right, points_single_1pix_up, points_single_1pix_left, points_single_1pix_down
        ]
    )

    no_rotation = Float64[1;0;;0;1;;]
    rotation_90_deg = Float64[0;1;;-1;0;;]

    no_translation = zeros(2)
    translation_halfpix_right = [0.0, 0.2]
    translation_1pix_down = [0.4, 0.0]

    zero_background = 0.0
    weight = 4.0

    # -------- interpolations ---------

    out = raster(grid_size, points_single_center, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 4 0 0
        0 0 0 0 0
        0 0 0 0 0
    ]
    
    out = raster(grid_size, points_single_1pix_right, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 4 0
        0 0 0 0 0
        0 0 0 0 0
    ]

    out = raster(grid_size, points_single_halfpix_down, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 2 0 0
        0 0 2 0 0
        0 0 0 0 0
    ]

    out = raster(grid_size, points_single_halfpix_down_and_right, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 1 1 0
        0 0 1 1 0
        0 0 0 0 0
    ]

    # -------- translations ---------

    out = raster(grid_size, points_four_cross, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 4 0 0
        0 4 0 4 0
        0 0 4 0 0
        0 0 0 0 0
    ]

    out = raster(grid_size, points_four_cross, no_rotation, translation_halfpix_right, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 2 2 0
        0 2 2 2 2
        0 0 2 2 0
        0 0 0 0 0
    ]

    out = raster(grid_size, points_four_cross, no_rotation, translation_1pix_down, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 4 0 0
        0 4 0 4 0
        0 0 4 0 0
    ]

    # -------- rotations ---------

    out = raster(grid_size, points_single_1pix_right, rotation_90_deg, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 4 0 0
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0
    ]
end


@testitem "raster inference and allocations" begin
    using BenchmarkTools, CUDA, StaticArrays
    include("../test/data.jl")

    # check type stability

    # single image
    @inferred DiffPointRasterisation.raster(D.grid_size_3d, D.points_static, D.rotation, D.translation_3d)
    @inferred DiffPointRasterisation.raster(D.grid_size_2d, D.points_static, D.projection, D.translation_2d)

    # batched canonical
    @inferred DiffPointRasterisation.raster(D.grid_size_3d, D.points_static, D.rotations_static, D.translations_3d_static)
    @inferred DiffPointRasterisation.raster(D.grid_size_2d, D.points_static, D.projections_static, D.translations_2d_static)

    # batched reinterpret reshape
    @inferred DiffPointRasterisation.raster(D.grid_size_3d, D.points_reinterp, D.rotations_reinterp, D.translations_3d_reinterp)
    @inferred DiffPointRasterisation.raster(D.grid_size_2d, D.points_reinterp, D.projections_reinterp, D.translations_2d_reinterp)
    if CUDA.functional()
        # single image
        @inferred DiffPointRasterisation.raster(D.grid_size_3d, cu(D.points_static), cu(D.rotation), cu(D.translation_3d))
        @inferred DiffPointRasterisation.raster(D.grid_size_2d, cu(D.points_static), cu(D.projection), cu(D.translation_2d))

        # batched
        @inferred DiffPointRasterisation.raster(D.grid_size_3d, cu(D.points_static), cu(D.rotations_static), cu(D.translations_3d_static))
        @inferred DiffPointRasterisation.raster(D.grid_size_2d, cu(D.points_static), cu(D.projections_static), cu(D.translations_2d_static))
    end

    # Ideally the sinlge image (non batched) case would be allocation-free.
    # The switch to KernelAbstractions made this allocating.
    # set test to broken for now.
    out_3d = Array{Float64, 3}(undef, D.grid_size_3d...)
    out_2d = Array{Float64, 2}(undef, D.grid_size_2d...)
    allocations = @ballocated DiffPointRasterisation.raster!($out_3d, $D.points_static, $D.rotation, $D.translation_3d) evals=1 samples=1
    @test allocations == 0 broken=true
    allocations = @ballocated DiffPointRasterisation.raster!($out_2d, $D.points_static, $D.projection, $D.translation_2d) evals=1 samples=1
    @test allocations == 0 broken=true
end


@testitem "raster batched consistency" begin
    include("../test/data.jl")

    #  raster
    out_3d = zeros(D.grid_size_3d..., D.batch_size)
    out_3d_batched = zeros(D.grid_size_3d..., D.batch_size)

    for (out_i, args...) in zip(eachslice(out_3d, dims=4), D.rotations, D.translations_3d, D.backgrounds, D.weights)
        raster!(out_i, D.more_points, args...)
    end

    DiffPointRasterisation.raster!(out_3d_batched, D.more_points, D.rotations, D.translations_3d, D.backgrounds, D.weights)

    #  raster_project
    out_2d = zeros(D.grid_size_2d..., D.batch_size)
    out_2d_batched = zeros(D.grid_size_2d..., D.batch_size)

    for (out_i, args...) in zip(eachslice(out_2d, dims=3), D.projections, D.translations_2d, D.backgrounds, D.weights)
        DiffPointRasterisation.raster!(out_i, D.more_points, args...)
    end

    DiffPointRasterisation.raster!(out_2d_batched, D.more_points, D.projections, D.translations_2d, D.backgrounds, D.weights)

    @test out_2d_batched ≈ out_2d
end