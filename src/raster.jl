"""
    raster(grid_size, points, rotation, translation, [background, weight])

Interpolate points (multi-) linearly into an Nd-array of size `grid_size`.

Before `points` are interpolated into the array, each point ``p`` is first
transformed according to
```math
\\hat{p} = R p + t
```
with `rotation` ``R`` and `translation` ``t``.

Points ``\\hat{p}`` that fall into the N-dimensional hypercube
with edges spanning from (-1, 1) in each dimension, are interpolated
into the output array.

The total `weight` of each point is distributed onto the 2^N nearest
pixels/voxels of the output array (according to the closeness of the
voxel center to the coordinates of point ``\\hat{p}``) via
N-linear interpolation.

`rotation`, `translation`, `background` and `weight` can have an 
additional "batch" dimension (as last dimension, and the axis
along this dimension must agree across the four arguments).
In this case, the output will also have that additional dimension.
This is useful if the same scene/points should be rastered from
different perspectives. 
"""
function raster end

# single image
raster(
    grid_size,
    points::AbstractMatrix{T},
    rotation::AbstractMatrix{<:Number},
    translation,
    background=zero(T),
    weight=one(T),
) where {T} = raster!(
    similar(points, grid_size),
    points,
    rotation,
    translation,
    background,
    weight,
)

# batched version
raster(
    grid_size,
    points::AbstractMatrix{T},
    rotation::AbstractArray{<:Number, 3},
    translation,
    background=Zeros(T, size(rotation, 3)),
    weight=Ones(T, size(rotation, 3)),
) where {T} = raster!(
    similar(points, (grid_size..., size(rotation, 3))),
    points,
    rotation,
    translation,
    background,
    weight,
)

@testitem "raster correctness" begin
    using Rotations
    grid_size = (5, 5)

    points_single_center = zeros(2, 1) 
    points_single_1pix_right = [0.0;0.4;;]
    points_single_1pix_up = [-0.4;0.0;;]
    points_single_1pix_left = [0.0;-0.4;;]
    points_single_1pix_down = [0.4;0.0;;]
    points_single_halfpix_down = [0.2;0.0;;]
    points_single_halfpix_down_and_right = [0.2;0.2;;]
    points_four_cross = reduce(
        hcat,
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


"""
    raster_project(grid_size, points, rotation, translation, [background, weight])

Interpolate N-dimensional points (multi-) linearly into an N-1 dimensional-array
of size `grid_size`.

Before `points` are interpolated into the array, each point ``p`` is first
transformed according to
```math
\\hat{p} = P R p + t
```

The remaining behaviour is the same as for `raster`
"""
function raster_project end

# single image
raster_project(
    grid_size,
    points::AbstractMatrix{T},
    rotation::AbstractMatrix{<:Number},
    translation,
    background=zero(T),
    weight=one(T),
) where {T} = raster_project!(
    similar(points, grid_size),
    points,
    rotation,
    translation,
    background,
    weight,
)

# batched version
raster_project(
    grid_size,
    points::AbstractMatrix{T},
    rotation::AbstractArray{<:Number, 3},
    translation,
    background=Zeros(T, size(rotation, 3)),
    weight=Ones(T, size(rotation, 3)),
) where {T} = raster_project!(
    similar(points, (grid_size..., size(rotation, 3))),
    points,
    rotation,
    translation,
    background,
    weight,
)


"""
    raster!(out, points, rotation, translation, [background, weight])

Inplace version of `raster`.

Write output into `out` and return `out`.
"""
raster!(
    out::AbstractArray{T, N_out},
    points,
    rotation::AbstractArray{<:Number, N_rotation},
    translation,
    background=N_rotation == 2 ? zero(T) : Zeros(T, size(rotation, 3)),
    weight=N_rotation == 2 ? one(T) : Ones(T, size(rotation, 3)),
) where {N_out, N_rotation, T<:Number} = _raster!(
    Val(N_out - (N_rotation - 2)),
    out,
    points,
    rotation,
    translation,
    background,
    weight,
)


@testitem "raster inference and allocations" begin
    using BenchmarkTools, CUDA
    include("../test/data.jl")

    # check type stability
    # single image
    @inferred DiffPointRasterisation.raster(D.grid_size_3d, D.points, D.rotation, D.translation_3d)
    @inferred DiffPointRasterisation.raster_project(D.grid_size_2d, D.points, D.rotation, D.translation_2d)
    # batched
    @inferred DiffPointRasterisation.raster(D.grid_size_3d, D.points, D.rotations, D.translations_3d)
    @inferred DiffPointRasterisation.raster_project(D.grid_size_2d, D.points, D.rotations, D.translations_2d)
    if CUDA.functional()
        @inferred DiffPointRasterisation.raster(D.grid_size_3d, cu(D.points), cu(D.rotation), cu(D.translation_3d))
        @inferred DiffPointRasterisation.raster_project(D.grid_size_2d, cu(D.points), cu(D.rotation), cu(D.translation_2d))
        # batched
        @inferred DiffPointRasterisation.raster(D.grid_size_3d, cu(D.points), cu(D.rotations), cu(D.translations_3d))
        @inferred DiffPointRasterisation.raster_project(D.grid_size_2d, cu(D.points), cu(D.rotations), cu(D.translations_2d))
    end

    # Ideally the sinlge image (non batched) case would be allocation-free.
    # The switch to KernelAbstractions made this allocating.
    # set test to broken for now.
    out_3d = Array{Float64, 3}(undef, D.grid_size_3d...)
    out_2d = Array{Float64, 2}(undef, D.grid_size_2d...)
    allocations = @ballocated DiffPointRasterisation.raster!($out_3d, $D.points, $D.rotation, $D.translation_3d) evals=1 samples=1
    @test allocations == 0 broken=true
    allocations = @ballocated DiffPointRasterisation.raster_project!($out_2d, $D.points, $D.rotation, $D.translation_2d) evals=1 samples=1
    @test allocations == 0 broken=true
end


"""
    raster_project!(out, points, rotation, translation, [background, weight])

Inplace version of `raster_project`.

Write output into `out` and return `out`.
"""
raster_project!(
    out::AbstractArray{T, N_out},
    points,
    rotation::AbstractArray{<:Number, N_rotation},
    translation,
    background=N_rotation == 2 ? zero(T) : Zeros(T, size(rotation, 3)),
    weight=N_rotation == 2 ? one(T) : Ones(T, size(rotation, 3)),
) where {N_out, N_rotation, T<:Number} = _raster!(
    Val(N_out + 1- (N_rotation - 2)),
    out,
    points,
    rotation,
    translation,
    background,
    weight,
)


_raster!(
    ::Val{N_in},
    out::AbstractArray{T, N_out},
    points::AbstractMatrix{<:Number},
    rotation::AbstractMatrix{<:Number},
    translation::AbstractVector{<:Number},
    background::Number,
    weight::Number,
) where {N_in, N_out, T<:Number} = drop_last_dim(
    _raster!(
        Val(N_in),
        append_singleton_dim(out),
        points,
        append_singleton_dim(rotation),
        append_singleton_dim(translation),
        fill!(similar(out, 1), background),
        fill!(similar(out, 1), weight),
    )
)


function _raster!(
    ::Val{N_in},
    out::AbstractArray{T, N_out_p1},
    points::AbstractMatrix{<:Number},
    rotation::AbstractArray{<:Number, 3},
    translation::AbstractMatrix{<:Number},
    background::AbstractVector{<:Number},
    weight::AbstractVector{<:Number},
) where {T<:Number, N_in, N_out_p1}
    N_out = N_out_p1 - 1
    out_batch_dim = ndims(out)
    batch_size = size(out, out_batch_dim)
    @argcheck size(points, 1) == size(rotation, 1) == size(rotation, 2) == N_in
    @argcheck batch_size == size(rotation, 3) == size(translation, 2) == size(background, 1) == size(weight, 1)
    @argcheck size(translation, 1) == N_out
    n_points = size(points, 2)

    scale = SVector{N_out, T}(size(out)[1:end-1]) / T(2)
    projection_idxs = SVector{N_out}(ntuple(identity, N_out))
    shifts=voxel_shifts(Val(N_out))
    out .= reshape(background, ntuple(_ -> 1, N_out)..., length(background))
    args = (Val(N_in), out, points, rotation, translation, weight, shifts, scale, projection_idxs)
    backend = get_backend(out)
    ndrange = (2^N_out, n_points, batch_size)
    workgroup_size = 1024 
    raster_kernel!(backend, workgroup_size, ndrange)(args...)
    out
end

@kernel function raster_kernel!(::Val{N_in}, out::AbstractArray{T, N_out_p1}, points, rotations, translations, weights, shifts, scale, projection_idxs) where {T, N_in, N_out_p1}
    N_out = N_out_p1 - 1  # dimensionality of output, without batch dimension
    neighbor_voxel_id, point_idx, batch_idx = @index(Global, NTuple)

    point = @inbounds SVector{N_in, T}(@view points[:, point_idx])
    rotation = @inbounds SMatrix{N_in, N_in, T}(@view rotations[:, :, batch_idx])
    translation = @inbounds SVector{N_out, T}(@view translations[:, batch_idx])
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

@testitem "raster batched consistency" begin
    include("../test/data.jl")

    #  raster
    out_3d = zeros(D.grid_size_3d..., D.batch_size)
    out_3d_batched = zeros(D.grid_size_3d..., D.batch_size)

    for (out_i, args...) in zip(eachslice(out_3d, dims=4), eachslice(D.rotations, dims=3), eachcol(D.translations_3d), D.backgrounds, D.weights)
        raster!(out_i, D.more_points, args...)
    end

    DiffPointRasterisation.raster!(out_3d_batched, D.more_points, D.rotations, D.translations_3d, D.backgrounds, D.weights)

    #  raster_project
    out_2d = zeros(D.grid_size_2d..., D.batch_size)
    out_2d_batched = zeros(D.grid_size_2d..., D.batch_size)

    for (out_i, args...) in zip(eachslice(out_2d, dims=3), eachslice(D.rotations, dims=3), eachcol(D.translations_2d), D.backgrounds, D.weights)
        DiffPointRasterisation.raster_project!(out_i, D.more_points, args...)
    end

    DiffPointRasterisation.raster_project!(out_2d_batched, D.more_points, D.rotations, D.translations_2d, D.backgrounds, D.weights)

    @test out_2d_batched ≈ out_2d
end