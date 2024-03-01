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

###############################################
# Step 1: Allocate output
###############################################

function raster(
    grid_size::Tuple,
    args...,
)
    eltypes = deep_eltype.(args)
    T = promote_type(eltypes...)
    rotation = args[2]
    if isa(rotation, AbstractMatrix)
        # non-batched
        out = similar(rotation, T, grid_size)
    else
        # batched
        @assert rotation isa AbstractVector{<:AbstractMatrix}
        batch_size = length(rotation)
        out = similar(rotation, T, (grid_size..., batch_size))
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
    points::AbstractVecOrMat,
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

raster!(out::AbstractArray{<:Number}, args::Vararg{AbstractVector, 5}) = raster!(out, canonical_arg.(args)...)

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


default_background(rotation::AbstractMatrix, T=eltype(rotation)) = zero(T)

default_background(rotation::AbstractVector{<:AbstractMatrix}, T=eltype(eltype(rotation))) = Zeros(T, length(rotation))

default_background(rotation::AbstractArray{_T, 3} where _T, T=eltype(rotation)) = Zeros(T, size(rotation, 3))

default_weight(rotation::AbstractMatrix, T=eltype(rotation)) = one(T)

default_weight(rotation::AbstractVector{<:AbstractMatrix}, T=eltype(eltype(rotation))) = Ones(T, length(rotation))

default_weight(rotation::AbstractArray{_T, 3} where _T, T=eltype(rotation)) = Ones(T, size(rotation, 3))


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