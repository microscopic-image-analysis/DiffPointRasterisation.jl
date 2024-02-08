module DiffPointRasterisationCUDAExt

using ArgCheck
using CUDA
using DiffPointRasterisation
using StaticArrays
using TestItems
using Cthulhu

# DiffPointRasterisation._raster!(
#     ::Val{N_in},
#     out::CuArray{<:Number, N_out},
#     points::CuMatrix{<:Number},
#     rotation::CuMatrix{<:Number},
#     translation::CuVector{<:Number},
#     background::Number,
#     weight::Number,
# ) where {N_in, N_out} = dropdims(
#     DiffPointRasterisation._raster!(
#         Val(N_in),
#         append_singleton_dim(out),
#         points,
#         append_singleton_dim(rotation),
#         append_singleton_dim(translation),
#         CUDA.fill(background, 1),
#         CUDA.fill(weight, 1),
#     );
#     dims=N_out+1
# )

function DiffPointRasterisation._raster!(
    ::Val{N_in},
    out::CuArray{<:Number, N_out},
    points::CuMatrix{<:Number},
    rotation::CuMatrix{<:Number},
    translation::CuVector{<:Number},
    background::Number,
    weight::Number,
) where {N_in, N_out}
    method = "singlebatch"
    @show method, size(out), size(rotation)
    dropdims(
        DiffPointRasterisation._raster!(
            Val(N_in),
            append_singleton_dim(out),
            points,
            append_singleton_dim(rotation),
            append_singleton_dim(translation),
            CUDA.fill(background, 1),
            CUDA.fill(weight, 1),
        );
        dims=N_out+1
    )
end

# function DiffPointRasterisation._raster!(
#     ::Val{N_in},
#     out::CuArray{T, N_out_p1},
#     points::CuMatrix{<:Number},
#     rotation::CuArray{<:Number, 3},
#     translation::CuMatrix{<:Number},
#     background::AbstractVector{<:Number},
#     weight::AbstractVector{<:Number},
# ) where {N_in, N_out_p1, T<:Number}
#     method = "multibatch"
#     @show method, size(out), size(rotation)
#     N_out = N_out_p1 - 1
#     @argcheck size(points, 1) == size(rotation, 1) == size(rotation, 2) == N_in
#     @argcheck size(translation, 1) == N_out
#     @argcheck size(out, ndims(out)) == size(rotation, 3) == size(translation, 2) == length(background) == length(weight)
# 
#     background, weight = cu.((background, weight))
# 
# 
#     scale = SVector{N_out, T}(size(out)[1:end-1]) / T(2)
#     shifts = DiffPointRasterisation.voxel_shifts(Val(N_out))
# 
#     out .= reshape(background, ntuple(_ -> 1, N_out)..., length(background))
#     n_points = size(points, 2)
#     batch_size = size(out, ndims(out))
#     point_coord_idxs = SVector{N_in}(ntuple(identity, N_in))
#     projection_idxs = SVector{N_out}(ntuple(identity, N_out))
# 
#     args = (Val(N_in), out, points, rotation, translation, weight, shifts, scale, point_coord_idxs, projection_idxs)
#     # return @device_code_warntype interactive=true @cuda launch=false raster_kernel!(args...)
#     let kernel = @cuda launch=false raster_kernel!(args...)
#         kernel_config = launch_configuration(kernel.fun)
#         available_threads = kernel_config.threads
#         @show available_threads
#         tx = 2^(N_out)
#         available_threads = available_threads ÷ tx
#         ty = min(n_points, available_threads)
#         available_threads = available_threads ÷ ty
#         tz = min(batch_size, available_threads)
# 
#         threads = (tx, ty, tz)
#         blocks = cld.((2^N_out, n_points, batch_size), threads)
#         @show threads
#         @show blocks
#         @show typeof.(args)
# 
#         kernel(args...; threads, blocks)
#     end
# 
#     out
# end

function DiffPointRasterisation._raster!(
    ::Val{N_in},
    out::CuArray{T, N_out_p1},
    points::CuMatrix{<:Number},
    rotation::CuArray{<:Number, 3},
    translation::CuMatrix{<:Number},
    background::AbstractVector{<:Number},
    weight::AbstractVector{<:Number},
) where {N_in, N_out_p1, T<:Number}
    method = "multibatch"
    @show method, size(out), size(rotation)
    N_out = N_out_p1 - 1
    @argcheck size(points, 1) == size(rotation, 1) == size(rotation, 2) == N_in
    @argcheck size(translation, 1) == N_out
    @argcheck size(out, ndims(out)) == size(rotation, 3) == size(translation, 2) == length(background) == length(weight)

    background, weight = cu.((background, weight))


    scale = SVector{N_out, T}(size(out)[1:end-1]) / T(2)
    shifts = DiffPointRasterisation.voxel_shifts(Val(N_out))

    out .= reshape(background, ntuple(_ -> 1, N_out)..., length(background))
    n_points = size(points, 2)
    batch_size = size(out, ndims(out))

    args = (Val(N_in), out, points, rotation, translation, weight, shifts, scale)
    # return @device_code_warntype interactive=true @cuda launch=false raster_kernel!(args...)
    let kernel = @cuda launch=false raster_kernel!(args...)
        kernel_config = launch_configuration(kernel.fun)
        available_threads = kernel_config.threads
        @show available_threads
        tx = 2^(N_out)
        available_threads = available_threads ÷ tx
        ty = min(n_points, available_threads)
        available_threads = available_threads ÷ ty
        tz = min(batch_size, available_threads)

        threads = (tx, ty, tz)
        blocks = cld.((2^N_out, n_points, batch_size), threads)

        kernel(args...; threads, blocks)
    end

    out
end

function raster_kernel!(
    ::Val{N_in},
    out::AbstractArray{T, N_out_p1},
    points::AbstractMatrix,
    rotations::AbstractArray{<:Number, 3},
    translations::AbstractMatrix,
    weights::AbstractVector,
    shifts::Tuple,
    scale::SVector,
) where {N_in, N_out_p1, T}
    N_out = N_out_p1 - 1
    neighbor_voxel_id = threadIdx().x
    point_idx = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    batch_idx = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if (batch_idx <= length(weights)) && (point_idx <= size(points, 2))
        rotview = view(rotations, :, :, batch_idx)
        @inbounds rotation_stat = SMatrix{N_in, N_in, T}(rotview)

        transview = view(translations, :, batch_idx)
        @inbounds translation_stat = SVector{N_out, T}(transview)

        origin = (-@SVector ones(T, N_out)) - translation_stat

        pointview = view(points, :, point_idx)
        @inbounds point = SVector{N_in, T}(pointview)

        shift = shifts[neighbor_voxel_id]

        projected_point = (rotation_stat * point - origin) .* scale
        idx_lower = CartesianIndex(Tuple(round.(Int, projected_point .- T(0.5), RoundUp)))

        voxel_idx = CartesianIndex(idx_lower + CartesianIndex(shift), batch_idx)
        if voxel_idx in CartesianIndices(out)
            val = sum(projected_point)
            @inbounds voxel_idx_linear = LinearIndices(out)[voxel_idx]
            @inbounds CUDA.@atomic out[voxel_idx_linear] += val
        end
    end

    nothing
end

# function raster_kernel!(
#     ::Val{N_in},
#     out::AbstractArray{T, N_out_p1},
#     points,
#     rotations,
#     translations,
#     weights,
#     shifts,
#     scale,
#     point_coord_idxs,
#     projection_idxs
# ) where {N_in, N_out_p1, T}
#     N_out = N_out_p1 - 1
#     neighbor_voxel_id = threadIdx().x
#     point_idx = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
#     point_stride = blockDim().y * gridDim().y 
#     batch_idx = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
#     batch_stride = blockDim().z * gridDim().z 
# 
#     @inbounds while batch_idx <= length(weights)
#         rotview = view(rotations, :, :, batch_idx)
#         mtype = SMatrix{N_in, N_in, T}
#         rotation_stat = @inline convert(mtype, rotview)
# 
#         # rotation_stat = @inbounds rotations[point_coord_idxs, point_coord_idxs, batch_idx]
#         translation_stat = @inbounds translations[point_coord_idxs, batch_idx]
# 
#         origin = (-@SVector ones(T, N_out)) - translation_stat
#         wgt = T(weights[batch_idx])
#         while point_idx <= size(points, 2)
#             point::SVector{N_in} = points[point_coord_idxs, point_idx]  # to get a SVector
#             shift = shifts[neighbor_voxel_id]
#             idx_lower, deltas = DiffPointRasterisation.raster_kernel(point, rotation_stat, origin, scale, projection_idxs)
# 
#             voxel_idx = CartesianIndex(idx_lower + CartesianIndex(shift), batch_idx)
#             voxel_idx in CartesianIndices(out) || continue
# 
#             val = DiffPointRasterisation.voxel_weight(deltas, shift, projection_idxs, wgt)
# 
#             voxel_idx_linear = LinearIndices(out)[voxel_idx]
#             CUDA.@atomic out[voxel_idx_linear] += val
# 
#             point_idx += point_stride
#         end
#         batch_idx += batch_stride
#     end
# 
#     nothing
# end


function DiffPointRasterisation.staticarrays()
    function canonical_kernel(::Val{N}, out, array3, matrix, vector) where {N}
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        rotation = view(array3, :, :, idx)
        column = view(matrix, :, idx)
        for i in 1:N 
            val = vector[i]
            for j in 1:N
                val += rotation[i, j] * column[j]
            end
            CUDA.@atomic out[i] += val
        end
        nothing
    end

    function sa_kernel(::Val{N}, out, array3, matrix, vector) where {N}
        idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        column = view(matrix, :, idx)
        rotation = view(array3, :, :, idx)
        @inbounds column_static = SVector{N}(column)  # @inbounds is required!
        @inbounds rotation_static = SMatrix{N, N}(rotation)  # @inbounds is required!
        val = rotation_static * column_static + vector
        for i in 1:N 
            val_i = val[i]
            CUDA.@atomic out[i] += val_i
        end
        nothing
    end

    N = 3
    S = 10
    svector = @SVector randn(Float32, N)
    vector = cu(svector)
    matrix = CUDA.randn(N, S)
    array3 = CUDA.randn(N, N, S)
    out_sa = fill!(similar(matrix, N), 0)
    out_canonical = fill!(similar(matrix, N), 0)

    canonical_args = (Val(N), out_canonical, array3, matrix, vector)
    sa_args = (Val(N), out_sa, array3, matrix, svector)

    let kernel = @cuda launch=false sa_kernel(sa_args...)
        available_threads = launch_configuration(kernel.fun).threads

        threads = min(S, available_threads)
        blocks = cld(S, threads)

        kernel(sa_args...; threads, blocks)
    end

    let kernel = @cuda launch=false canonical_kernel(canonical_args...)
        available_threads = launch_configuration(kernel.fun).threads

        threads = min(S, available_threads)
        blocks = cld(S, threads)

        kernel(canonical_args...; threads, blocks)
    end

    out_sa, out_canonical
end

@inline append_singleton_dim(a) = reshape(a, size(a)..., 1)

@inline drop_last_dim(a) = dropdims(a; dims=ndims(a))

end  # module DiffPointRasterisationCUDAExt