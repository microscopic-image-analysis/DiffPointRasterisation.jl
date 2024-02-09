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
"""
function raster end

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

@testitem "raster single point 2D" begin
    using Rotations, StaticArrays
    grid_size = (5, 5)

    points_single_center = zeros(2, 1) 
    no_rotation = Float64[1;0;;0;1;;]
    no_translation = zeros(2)
    zero_background = 0.0
    weight = 4.0

    out = raster(grid_size, points_single_center, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 4 0 0
        0 0 0 0 0
        0 0 0 0 0
    ]
    
    points_single_1pix_right = [0.0;0.4;;]
    out = raster(grid_size, points_single_1pix_right, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 4 0
        0 0 0 0 0
        0 0 0 0 0
    ]

    points_single_halfpix_down = [0.2;0.0;;]
    out = raster(grid_size, points_single_halfpix_down, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 2 0 0
        0 0 2 0 0
        0 0 0 0 0
    ]

    points_single_halfpix_down_and_right = [0.2;0.2;;]
    out = raster(grid_size, points_single_halfpix_down_and_right, no_rotation, no_translation, zero_background, weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 1 1 0
        0 0 1 1 0
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
with N-dimensional `rotation` ``R``, projection ``P`` and N-1-dmensional
`translation` ``t``.
The projection simply drops the last coordinate of ``R p``.

Points ``\\hat{p}`` that fall into the N-1-dimensional hypercube
with edges spanning from (-1, 1) in each dimension, are interpolated
into the output array.

The total `weight` of each point is distributed onto the 2^(N-1) nearest
pixels/voxels of the output array (according to the closeness of the
voxel center to the coordinates of point ``\\hat{p}``) via
N-1-linear interpolation.
"""
function raster_project end

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


@testitem "raster! allocations" begin
    using BenchmarkTools, Rotations
    out = zeros(8, 8, 8)
    points = randn(3, 10)
    rotation = rand(QuatRotation)
    translation = zeros(3)

    allocations = @ballocated DiffPointRasterisation.raster!($out, $points, $rotation, $translation) evals=1 samples=1
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

@testitem "raster_project! allocations" begin
    using BenchmarkTools, Rotations
    out = zeros(16, 16)
    points = randn(3, 10)
    rotation = rand(QuatRotation)
    translation = zeros(2)

    allocations = @ballocated DiffPointRasterisation.raster_project!($out, $points, $rotation, $translation) evals=1 samples=1
    @test allocations == 0 broken=true
end


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
    raster_kernel!(backend, 64)(args...; ndrange=(2^N_out, n_points, batch_size))
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
    # voxel filled by this kernel instance is
    # coordinate of reference voxel plus shift
    voxel_idx = CartesianIndex(CartesianIndex(Tuple(coord_reference_voxel)) + CartesianIndex(shift), batch_idx)

    if voxel_idx in CartesianIndices(out)
        val = voxel_weight(deltas, shift, projection_idxs, weight)
        @inbounds Atomix.@atomic out[voxel_idx] += val
    end
    nothing
end

@inline function voxel_weight(deltas, shift, projection_idxs, point_weight)
    lower_upper = mod1.(shift, 2)
    delta_idxs = CartesianIndex.(projection_idxs, lower_upper)
    val = prod(@inbounds @view deltas[delta_idxs]) * point_weight
    val
end

@testitem "raster! batched" begin
    using BenchmarkTools, Rotations
    include("testing.jl")

    batch_size = batch_size_for_test()

    out = zeros(8, 8, 8, batch_size)
    out_batched = zeros(8, 8, 8, batch_size)
    points = 0.3 .* randn(3, 10)
    rotation = stack(rand(QuatRotation, batch_size))
    translation = zeros(3, batch_size)
    background = zeros(batch_size)
    weight = ones(batch_size)

    for (out_i, args...) in zip(eachslice(out, dims=4), eachslice(rotation, dims=3), eachcol(translation), background, weight)
        raster!(out_i, points, args...)
    end

    DiffPointRasterisation.raster!(out_batched, points, rotation, translation, background, weight)

    @test out_batched ≈ out
end

@testitem "raster_project! batched" begin
    using BenchmarkTools, Rotations
    include("testing.jl")

    batch_size = batch_size_for_test()

    out = zeros(16, 16, batch_size)
    out_batched = zeros(16, 16, batch_size)
    points = 0.3 .* randn(3, 10)
    rotation = stack(rand(QuatRotation, batch_size))
    translation = zeros(2, batch_size)
    background = zeros(batch_size)
    weight = ones(batch_size)

    for (out_i, args...) in zip(eachslice(out, dims=3), eachslice(rotation, dims=3), eachcol(translation), background, weight)
        DiffPointRasterisation.raster_project!(out_i, points, args...)
    end

    DiffPointRasterisation.raster_project!(out_batched, points, rotation, translation, background, weight)

    @test out_batched ≈ out
end