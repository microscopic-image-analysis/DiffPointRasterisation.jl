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
        batch_size = isa(rotation, AbstractVector) ? length(rotation) : size(rotation, 3)
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
# Step 3: Convert arguments to canonical from
#         i.e. vec-of-array style
###############################################

#----------------------------------------------
# Step 3a: If input is for a single image
#----------------------------------------------

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
        append_singleton_dim(rotation),
        append_singleton_dim(translation),
        append_singleton_dim(background),
        append_singleton_dim(weight),
    )
)

#----------------------------------------------
# Step 3b: If input is for a batch of images
#----------------------------------------------

raster!(out::AbstractArray{<:Number}, args::Vararg{Any, 5}) = raster!(out, canonical_arg.(args)...)

###############################################
# Step 4: Error on inconsistent dimensions
###############################################

# if N_out_rot == N_out_trans this should not be called
# because the actual implementation specializes on N_out
raster!(
    ::AbstractArray{<:Number},
    ::AbstractVector{<:StaticVector{N_in, <:Number}},
    ::AbstractVector{<:StaticMatrix{N_out_rot, N_in, <:Number}},
    ::AbstractVector{<:StaticVector{N_out_trans, <:Number}},
    ::AbstractVector{<:Number},
    ::AbstractVector{<:Number},
) where {N_in, N_out_rot, N_out_trans} = error("Row dimension of rotation (got $N_out_rot) and translation (got $N_out_trans) must agree!")


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
        @testset "canonical arguments (vec of array)" begin
            out = raster(
                D.grid_size_3d,
                D.points_vec,
                D.rotations_vec,
                D.translations_3d_vec,
                D.backgrounds,
                D.weights
            )
        end
        @testset "points as matrix" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points,
                D.rotations_vec,
                D.translations_3d_vec,
                D.backgrounds,
                D.weights
            )
        end
        @testset "rotations as 3d-array" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points_vec,
                D.rotations,
                D.translations_3d_vec,
                D.backgrounds,
                D.weights
            )
        end
        @testset "translations as matrix" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points_vec,
                D.rotations_vec,
                D.translations_3d,
                D.backgrounds,
                D.weights
            )
        end
        @testset "all as nd-array" begin
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
            D.points_vec,
            D.rotations_vec,
            D.translations_3d_vec,
            zeros(D.batch_size),
            ones(D.batch_size),
        )
        @testset "default argmuments canonical" begin
            @test out ≈ raster(
                D.grid_size_3d,
                D.points_vec,
                D.rotations_vec,
                D.translations_3d_vec,
            )
        end
        @testset "default arguments all as nd-array" begin
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
        @testset "canonical arguments (vec of array)" begin
            out = raster(
                D.grid_size_2d,
                D.points_vec,
                D.projections_vec,
                D.translations_2d_vec,
                D.backgrounds,
                D.weights
            )
        end
        @testset "points as matrix" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points,
                D.projections_vec,
                D.translations_2d_vec,
                D.backgrounds,
                D.weights
            )
        end
        @testset "projections as 3d-array" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_vec,
                D.projections,
                D.translations_2d_vec,
                D.backgrounds,
                D.weights
            )
        end
        @testset "translations as matrix" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_vec,
                D.projections_vec,
                D.translations_2d,
                D.backgrounds,
                D.weights
            )
        end
        @testset "all as nd-array" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_vec,
                D.projections,
                D.translations_2d,
                D.backgrounds,
                D.weights
            )
        end
        out = raster(
            D.grid_size_2d,
            D.points_vec,
            D.projections_vec,
            D.translations_2d_vec,
            zeros(D.batch_size),
            ones(D.batch_size),
        )
        @testset "default argmuments canonical" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points_vec,
                D.projections_vec,
                D.translations_2d_vec,
            )
        end
        @testset "default arguments all as nd-array" begin
            @test out ≈ raster(
                D.grid_size_2d,
                D.points,
                D.projections,
                D.translations_2d,
            )
        end
    end
end