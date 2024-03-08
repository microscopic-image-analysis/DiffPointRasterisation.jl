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

# Arguments
- `grid_size`: Tuple of integers defining the output dimensions
- `points::AbstractVector{<:AbstractVector}`: A vector of same length
  vectors representing points
- `rotation`: Either a single matrix(-like object) or a vector of such,
  that linearly transform(s) `points` before rasterisation.
- `translation`: Either a single vector or a vector of such, that
  translates `points` *after* `rotation`. If `rotation` includes a
  projection, `translation` thus needs to have the same length as
  `rotation * points[i]`.
- `background`: Either a single number or a vector of such.
- `weight`: Either a single number or a vector of such.

`rotation`, `translation`, `background` and `weight` can have an 
additional "batch" dimension (by providing them as vectors of single
parameters. The length of these vectors must be the same for all four
arguments).
In this case, the output array will have dimensionality +1 with an
additional axis on last position corresponding to the number of
elements in the batch.
See [Raster a single point cloud to a batch of poses](@ref) for more
details.

See also: [`raster!`](@ref)
"""
function raster end

"""
    raster!(out, points, rotation, translation, [background, weight])

Interpolate points (multi-) linearly into the Nd-array `out`.
In-place version of [`raster`](@ref). See there for details.
"""
function raster! end

###############################################
# Step 1: Allocate output
###############################################

function raster(
    grid_size::Tuple,
    args...,
)
    eltypes = deep_eltype.(args)
    T = promote_type(eltypes...)
    points = args[1]
    rotation = args[2]
    if isa(rotation, AbstractMatrix)
        # non-batched
        out = similar(points, T, grid_size)
    else
        # batched
        @assert rotation isa AbstractVector{<:AbstractMatrix}
        batch_size = length(rotation)
        out = similar(points, T, (grid_size..., batch_size))
    end
    raster!(out, args...)
end

deep_eltype(el) = deep_eltype(typeof(el))
deep_eltype(t::Type) = t
deep_eltype(t::Type{<:AbstractArray}) = deep_eltype(eltype(t))


###############################################
# Step 2: Fill default arguments if necessary
###############################################

@inline raster!(out::AbstractArray{<:Number}, args::Vararg{Any, 3}) = raster!(out, args..., default_background(args[2]))
@inline raster!(out::AbstractArray{<:Number}, args::Vararg{Any, 4}) = raster!(out, args..., default_weight(args[2]))

###############################################
# Step 3: Convenience interface for single image:
#         Convert arguments for single image to
#         length-1 vec of arguments
###############################################

raster!(
    out::AbstractArray{<:Number},
    points::AbstractVector{<:AbstractVector{<:Number}},
    rotation::AbstractMatrix{<:Number},
    translation::AbstractVector{<:Number},
    background::Number,
    weight::Number,
) = drop_last_dim(
    raster!(
        append_singleton_dim(out),
        points,
        @SVector([rotation]),
        @SVector([translation]),
        @SVector([background]),
        @SVector([weight]),
    )
)

###############################################
# Step 4: Convert arguments to canonical form,
#         i.e. vectors of statically sized arrays
###############################################

raster!(out::AbstractArray{<:Number}, args::Vararg{AbstractVector, 5}) = raster!(out, inner_to_sized.(args)...)

###############################################
# Step 5: Error on inconsistent dimensions
###############################################

# if N_out_rot == N_out_trans this should not be called
# because the actual implementation specializes on N_out
function raster!(
    ::AbstractArray{<:Number, N_out},
    ::AbstractVector{<:StaticVector{N_in, <:Number}},
    ::AbstractVector{<:StaticMatrix{N_out_rot, N_in_rot, <:Number}},
    ::AbstractVector{<:StaticVector{N_out_trans, <:Number}},
    ::AbstractVector{<:Number},
    ::AbstractVector{<:Number},
) where {N_in, N_out, N_in_rot, N_out_rot, N_out_trans}
    if N_out_trans != N_out
        error("Dimension of translation (got $N_out_trans) and output dimentsion (got $N_out) must agree!")
    end
    if N_out_rot != N_out
        error("Row dimension of rotation (got $N_out_rot) and output dimentsion (got $N_out) must agree!")
    end
    if N_in_rot != N_in
        error("Column dimension of rotation (got $N_in_rot) and points (got $N_in) must agree!")
    end
    error("Dispatch error. Should not arrive here. Please file a bug.")
end

# now similar for pullback
"""
    raster_pullback!(
        ds_dout, args...;
        [points, rotation, translation, background, weight]
    )

Pullback for [`raster`](@ref) / [`raster!`](@ref).

Take as input `ds_dout` the sensitivity of some quantity (`s` for "scalar")
to the *output* `out` of the function `out = raster(grid_size, args...)`
(or `out = raster!(out, args...)`), as well as
the exact same arguments `args` that were passed to `raster`/`raster!`, and
return the sensitivities of `s` to the *inputs* `args` of the function
`raster`/`raster!`.

Optionally, pre-allocated output arrays for each input sensitivity can be
specified as keyword arguments with the name of the original argument to
`raster` as key, and a nd-array as value, where the n-th dimension is the
batch dimension.
For example to provide a pre-allocated array for the sensitivity of `s` to
the `translation` argument of `raster`, do:
`sensitivities = raster_pullback!(ds_dout, args...; translation = [zeros(2) for _ in 1:8])`
for 2-dimensional points and a batch size of 8.
See also [Raster a single point cloud to a batch of poses](@ref)
"""
function raster_pullback! end


###############################################
# Step 1: Fill default arguments if necessary
###############################################

@inline raster_pullback!(ds_out::AbstractArray{<:Number}, args::Vararg{Any, 3}; kwargs...) = raster_pullback!(ds_out, args..., default_background(args[2]); kwargs...)
@inline raster_pullback!(ds_dout::AbstractArray{<:Number}, args::Vararg{Any, 4}; kwargs...) = raster_pullback!(ds_dout, args..., default_weight(args[2]); kwargs...)


###############################################
# Step 2: Convert arguments to canonical form,
#         i.e. vectors of statically sized arrays
###############################################

# single image
raster_pullback!(
    ds_dout::AbstractArray{<:Number},
    points::AbstractVector{<:AbstractVector{<:Number}},
    rotation::AbstractMatrix{<:Number},
    translation::AbstractVector{<:Number},
    background::Number,
    weight::Number;
    kwargs...
) = raster_pullback!(
    ds_dout,
    inner_to_sized(points),
    to_sized(rotation),
    to_sized(translation),
    background,
    weight;
    kwargs...
)

# batch of images
raster_pullback!(ds_dout::AbstractArray{<:Number}, args::Vararg{AbstractVector, 5}; kwargs...) = raster_pullback!(ds_dout, inner_to_sized.(args)...; kwargs...)


###############################################
# Step 3: Allocate output
###############################################

# single image
raster_pullback!(
    ds_dout::AbstractArray{<:Number, N_out},
    inp_points::AbstractVector{<:StaticVector{N_in, T}},
    inp_rotation::StaticMatrix{N_out, N_in, <:Number},
    inp_translation::StaticVector{N_out, <:Number},
    inp_background::Number,
    inp_weight::Number;
    points::AbstractMatrix{T} = default_ds_dpoints_single(inp_points, N_in),
    kwargs...
) where {N_in, N_out, T<:Number} = raster_pullback!(
    ds_dout,
    inp_points,
    inp_rotation,
    inp_translation,
    inp_background,
    inp_weight,
    points;
    kwargs...
)

# batch of images
raster_pullback!(
    ds_dout::AbstractArray{<:Number},
    inp_points::AbstractVector{<:StaticVector{N_in, TP}},
    inp_rotation::AbstractVector{<:StaticMatrix{N_out, N_in, TR}},
    inp_translation::AbstractVector{<:StaticVector{N_out, TT}},
    inp_background::AbstractVector{TB},
    inp_weight::AbstractVector{TW};
    points::AbstractArray{TP} = default_ds_dpoints_batched(inp_points, N_in, length(inp_rotation)),
    rotation::AbstractArray{TR, 3} = similar(inp_rotation, TR, (N_out, N_in, length(inp_rotation))),
    translation::AbstractMatrix{TT} = similar(inp_translation, TT, (N_out, length(inp_translation))),
    background::AbstractVector{TB} = similar(inp_background),
    weight::AbstractVector{TW} = similar(inp_weight),
) where {N_in, N_out, TP<:Number, TR<:Number, TT<:Number, TB<:Number, TW<:Number} = raster_pullback!(
    ds_dout,
    inp_points,
    inp_rotation,
    inp_translation,
    inp_background,
    inp_weight,
    points,
    rotation,
    translation,
    background,
    weight,
)


###############################################
# Step 4: Error on inconsistent dimensions
###############################################

# single image
raster_pullback!(
    ::AbstractArray{<:Number, N_out},
    ::AbstractVector{<:StaticVector{N_in, <:Number}},
    ::StaticMatrix{N_out_rot, N_in_rot, <:Number},
    ::StaticVector{N_out_trans, <:Number},
    ::Number,
    ::Number,
    ::AbstractMatrix{<:Number};
    kwargs...
) where {N_in, N_out, N_in_rot, N_out_rot, N_out_trans} = error_dimensions(
    N_in,
    N_out,
    N_in_rot,
    N_out_rot,
    N_out_trans
)

# batch of images
raster_pullback!(
    ::AbstractArray{<:Number, N_out_p1},
    ::AbstractVector{<:StaticVector{N_in, <:Number}},
    ::AbstractVector{<:StaticMatrix{N_out_rot, N_in_rot, <:Number}},
    ::AbstractVector{<:StaticVector{N_out_trans, <:Number}},
    ::AbstractVector{<:Number},
    ::AbstractVector{<:Number},
    ::AbstractArray{<:Number},
    ::AbstractArray{<:Number, 3},
    ::AbstractMatrix{<:Number},
    ::AbstractVector{<:Number},
    ::AbstractVector{<:Number},
) where {N_in, N_out_p1, N_in_rot, N_out_rot, N_out_trans} = error_dimensions(
    N_in,
    N_out_p1 - 1,
    N_in_rot,
    N_out_rot,
    N_out_trans
)

function error_dimensions(N_in, N_out, N_in_rot, N_out_rot, N_out_trans)
    if N_out_trans != N_out
        error("Dimension of translation (got $N_out_trans) and output dimentsion (got $N_out) must agree!")
    end
    if N_out_rot != N_out
        error("Row dimension of rotation (got $N_out_rot) and output dimentsion (got $N_out) must agree!")
    end
    if N_in_rot != N_in
        error("Column dimension of rotation (got $N_in_rot) and points (got $N_in) must agree!")
    end
    error("Dispatch error. Should not arrive here. Please file a bug.")
end

default_background(rotation::AbstractMatrix, T=eltype(rotation)) = zero(T)

default_background(rotation::AbstractVector{<:AbstractMatrix}, T=eltype(eltype(rotation))) = Zeros(T, length(rotation))

default_background(rotation::AbstractArray{_T, 3} where _T, T=eltype(rotation)) = Zeros(T, size(rotation, 3))

default_weight(rotation::AbstractMatrix, T=eltype(rotation)) = one(T)

default_weight(rotation::AbstractVector{<:AbstractMatrix}, T=eltype(eltype(rotation))) = Ones(T, length(rotation))

default_weight(rotation::AbstractArray{_T, 3} where _T, T=eltype(rotation)) = Ones(T, size(rotation, 3))

default_ds_dpoints_single(points::AbstractVector{<:AbstractVector{TP}}, N_in) where {TP<:Number} = similar(points, TP, (N_in, length(points)))

default_ds_dpoints_batched(points::AbstractVector{<:AbstractVector{TP}}, N_in, batch_size) where {TP<:Number} = similar(points, TP, (N_in, length(points), min(batch_size, Threads.nthreads())))


@testitem "raster interface" begin
    include("../test/data.jl")

    @testset "no projection" begin
        local out
        @testset "canonical arguments (vec of staticarray)" begin
            out = raster(
                D.grid_size_3d,
                D.points_static,
                D.rotations_static,
                D.translations_3d_static,
                D.backgrounds,
                D.weights
            )
        end
        @testset "reinterpret nd-array as vec-of-array" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points_reinterp,
                D.rotations_reinterp,
                D.translations_3d_reinterp,
                D.backgrounds,
                D.weights
            )
        end
        @testset "point as non-static vector" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points,
                D.rotations_static,
                D.translations_3d_static,
                D.backgrounds,
                D.weights
            )
        end
        @testset "rotation as non-static matrix" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points_static,
                D.rotations,
                D.translations_3d_static,
                D.backgrounds,
                D.weights
            )
        end
        @testset "translation as non-static vector" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points_static,
                D.rotations_static,
                D.translations_3d,
                D.backgrounds,
                D.weights
            )
        end
        @testset "all as non-static array" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points,
                D.rotations,
                D.translations_3d,
                D.backgrounds,
                D.weights
            )
        end
        out = raster(
            D.grid_size_3d,
            D.points_static,
            D.rotations_static,
            D.translations_3d_static,
            zeros(D.batch_size),
            ones(D.batch_size),
        )
        @testset "default argmuments canonical" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points_static,
                D.rotations_static,
                D.translations_3d_static,
            )
        end
        @testset "default arguments all as non-static array" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points,
                D.rotations,
                D.translations_3d,
            )
        end
    end

    @testset "projection" begin
        local out
        @testset "canonical arguments (vec of staticarray)" begin
            out = raster(
                D.grid_size_2d,
                D.points_static,
                D.projections_static,
                D.translations_2d_static,
                D.backgrounds,
                D.weights
            )
        end
        @testset "reinterpret nd-array as vec-of-array" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_reinterp,
                D.projections_reinterp,
                D.translations_2d_reinterp,
                D.backgrounds,
                D.weights
            )
        end
        @testset "point as non-static vector" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points,
                D.projections_static,
                D.translations_2d_static,
                D.backgrounds,
                D.weights
            )
        end
        @testset "projection as non-static matrix" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_static,
                D.projections,
                D.translations_2d_static,
                D.backgrounds,
                D.weights
            )
        end
        @testset "translation as non-static vector" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_static,
                D.projections_static,
                D.translations_2d,
                D.backgrounds,
                D.weights
            )
        end
        @testset "all as non-static array" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_static,
                D.projections,
                D.translations_2d,
                D.backgrounds,
                D.weights
            )
        end
        out = raster(
            D.grid_size_2d,
            D.points_static,
            D.projections_static,
            D.translations_2d_static,
            zeros(D.batch_size),
            ones(D.batch_size),
        )
        @testset "default argmuments canonical" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_static,
                D.projections_static,
                D.translations_2d_static,
            )
        end
        @testset "default arguments all as non-static array" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points,
                D.projections,
                D.translations_2d,
            )
        end
    end
end