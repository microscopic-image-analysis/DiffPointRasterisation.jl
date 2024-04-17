# We provide an explicit extension package for CUDA
# since the pullback kernel profits a lot from 
# parallel reductions, which are relatively straightforwadly
# expressed using while loops.
# However KernelAbstractions currently does not play nicely
# with while loops, see e.g. here:
# https://github.com/JuliaGPU/KernelAbstractions.jl/issues/330
module DiffPointRasterisationCUDAExt

using DiffPointRasterisation, CUDA
using ArgCheck
using FillArrays
using StaticArrays


const CuOrFillArray{T, N} = Union{CuArray{T, N}, FillArrays.AbstractFill{T, N}}


const CuOrFillVector{T} = CuOrFillArray{T, 1}


function raster_pullback_kernel!(
    ::Type{T},
    ds_dout,
    points::AbstractVector{<:StaticVector{N_in}},
    rotations::AbstractVector{<:StaticMatrix{N_out, N_in, TR}},
    translations::AbstractVector{<:StaticVector{N_out, TT}},
    out_weights,
    point_weights,
    shifts,
    scale,
    # outputs:
    ds_dpoints,
    ds_drotation,
    ds_dtranslation,
    ds_dout_weight,
    ds_dpoint_weight,

) where {T, TR, TT, N_in, N_out}
    n_voxel = blockDim().z
    points_per_workgroup = blockDim().x
    batchsize_per_workgroup = blockDim().y
    # @assert points_per_workgroup == 1
    # @assert n_voxel == 2^N_out
    # @assert threadIdx().x == 1
    n_threads_per_workgroup = n_voxel * batchsize_per_workgroup

    s = threadIdx().z
    b = threadIdx().y
    thread = (b - 1) * n_voxel + s

    neighbor_voxel_id = (blockIdx().z - 1) * n_voxel + s
    point_idx = (blockIdx().x - 1) * points_per_workgroup + threadIdx().x
    batch_idx = (blockIdx().y - 1) * batchsize_per_workgroup + b
    in_batch = batch_idx <= length(rotations)

    dimension1 = (N_out, n_voxel, batchsize_per_workgroup)
    ds_dpoint_rot_shared = CuDynamicSharedArray(T, dimension1)
    offset = sizeof(T) * prod(dimension1)
    dimension2 = (N_in, batchsize_per_workgroup)
    ds_dpoint_shared = CuDynamicSharedArray(T, dimension2, offset)
    dimension3 = (n_voxel, batchsize_per_workgroup)
    offset += sizeof(T) * prod(dimension2)
    ds_dpoint_weight_shared = CuDynamicSharedArray(T, dimension3, offset)

    rotation = @inbounds in_batch ? rotations[batch_idx] : @SMatrix zeros(TR, N_in, N_in)
    point = @inbounds points[point_idx]
    point_weight = @inbounds point_weights[point_idx]

    if in_batch
        translation = @inbounds translations[batch_idx]
        out_weight = @inbounds out_weights[batch_idx]
        shift = @inbounds shifts[neighbor_voxel_id]
        origin = (-@SVector ones(TT, N_out)) - translation

        coord_reference_voxel, deltas = DiffPointRasterisation.reference_coordinate_and_deltas(
            point,
            rotation,
            origin,
            scale,
        )
        voxel_idx = CartesianIndex(CartesianIndex(Tuple(coord_reference_voxel)) + CartesianIndex(shift), batch_idx)


        ds_dweight_local = zero(T)
        if voxel_idx in CartesianIndices(ds_dout)
            @inbounds ds_dweight_local = DiffPointRasterisation.voxel_weight(
                deltas,
                shift,
                ds_dout[voxel_idx],
            )

            factor = ds_dout[voxel_idx] * out_weight * point_weight
            ds_dcoord_part = SVector(factor .* ntuple(n -> DiffPointRasterisation.interpolation_weight(n, N_out, deltas, shift), Val(N_out)))
            @inbounds ds_dpoint_rot_shared[:, s, b] .= ds_dcoord_part .* scale
        else
            @inbounds ds_dpoint_rot_shared[:, s, b] .= zero(T)
        end

        @inbounds ds_dpoint_weight_shared[s, b] = ds_dweight_local * out_weight
        ds_dout_weight_local = ds_dweight_local * point_weight
        @inbounds CUDA.@atomic ds_dout_weight[batch_idx] += ds_dout_weight_local
    else
        @inbounds ds_dpoint_weight_shared[s, b] = zero(T)
        @inbounds ds_dpoint_rot_shared[:, s, b] .= zero(T)
    end

    # parallel summation of ds_dpoint_rot_shared over neighboring-voxel dimension
    # for a given thread-local batch index
    stride = 1
    @inbounds while stride < n_voxel
        sync_threads()
        idx = 2 * stride * (s - 1) + 1
        if idx <= n_voxel
            dim = 1
            while dim <= N_out
                other_val_p = if idx + stride <= n_voxel
                    ds_dpoint_rot_shared[dim, idx + stride, b]
                else
                    zero(T)
                end
                ds_dpoint_rot_shared[dim, idx, b] += other_val_p
                dim += 1
            end
        end
        stride *= 2
    end

    sync_threads()

    if in_batch
        dim = s
        if dim <= N_out
            coef = ds_dpoint_rot_shared[dim, 1, b]
            @inbounds CUDA.@atomic ds_dtranslation[dim, batch_idx] += coef
            j = 1
            while j <= N_in
                val = coef * point[j]
                @inbounds CUDA.@atomic ds_drotation[dim, j, batch_idx] += val 
                j += 1
            end
        end
    end

    # derivative of point with respect to rotation per batch dimension
    dim = s
    while dim <= N_in
        val = zero(T)
        j = 1
        while j <= N_out
            @inbounds val += rotation[j, dim] * ds_dpoint_rot_shared[j, 1, b]
            j += 1
        end
        @inbounds ds_dpoint_shared[dim, b] = val
        dim += n_voxel
    end

    # parallel summation of ds_dpoint_shared over batch dimension
    stride = 1
    @inbounds while stride < batchsize_per_workgroup
        sync_threads()
        idx = 2 * stride * (b - 1) + 1
        if idx <= batchsize_per_workgroup
            dim = s 
            while dim <= N_in
                other_val_p = if idx + stride <= batchsize_per_workgroup
                    ds_dpoint_shared[dim, idx + stride]
                else
                    zero(T)
                end
                ds_dpoint_shared[dim, idx] += other_val_p
                dim += n_voxel
            end
        end
        stride *= 2
    end

    # parallel summation of ds_dpoint_weight_shared over voxel and batch dimension
    stride = 1
    @inbounds while stride < n_threads_per_workgroup
        sync_threads()
        idx = 2 * stride * (thread - 1) + 1
        if idx <= n_threads_per_workgroup
            other_val_w = if idx + stride <= n_threads_per_workgroup 
                ds_dpoint_weight_shared[idx + stride]
            else
                zero(T)
            end
            ds_dpoint_weight_shared[idx] += other_val_w
        end
        stride *= 2
    end

    sync_threads()

    dim = thread
    while dim <= N_in
        val = ds_dpoint_shared[dim, 1]
        # batch might be split across blocks, so need atomic add
        @inbounds CUDA.@atomic ds_dpoints[dim, point_idx] += val
        dim += n_threads_per_workgroup
    end

    if thread == 1
        val_w = ds_dpoint_weight_shared[1, 1]
        # batch might be split across blocks, so need atomic add
        @inbounds CUDA.@atomic ds_dpoint_weight[point_idx] += val_w
    end

    nothing
end

# single image
raster_pullback!(
    ds_dout::CuArray{<:Number, N_out},
    points::AbstractVector{<:StaticVector{N_in, <:Number}},
    rotation::StaticMatrix{N_out, N_in, <:Number},
    translation::StaticVector{N_out, <:Number},
    background::Number,
    out_weight::Number,
    point_weight::CuOrFillVector{<:Number},
    ds_dpoints::AbstractMatrix{<:Number},
    ds_dpoint_weight::AbstractVector{<:Number};
    kwargs...
) where {N_in, N_out} = error("Not implemented: raster_pullback! for single image not implemented on GPU. Consider using CPU arrays")

# batch of images
function DiffPointRasterisation.raster_pullback!(
    ds_dout::CuArray{<:Number, N_out_p1},
    points::CuVector{<:StaticVector{N_in, <:Number}},
    rotation::CuVector{<:StaticMatrix{N_out, N_in, <:Number}},
    translation::CuVector{<:StaticVector{N_out, <:Number}},
    background::CuOrFillVector{<:Number},
    out_weight::CuOrFillVector{<:Number},
    point_weight::CuOrFillVector{<:Number},
    ds_dpoints::CuMatrix{TP},
    ds_drotation::CuArray{TR, 3},
    ds_dtranslation::CuMatrix{TT},
    ds_dbackground::CuVector{<:Number},
    ds_dout_weight::CuVector{OW},
    ds_dpoint_weight::CuVector{PW},
) where {N_in, N_out, N_out_p1, TP<:Number, TR<:Number, TT<:Number, OW<:Number, PW<:Number}
    T = promote_type(eltype(ds_dout), TP, TR, TT, OW, PW)
    batch_axis = axes(ds_dout, N_out_p1)
    @argcheck N_out == N_out_p1 - 1
    @argcheck batch_axis == axes(rotation, 1) == axes(translation, 1) == axes(background, 1) == axes(out_weight, 1)
    @argcheck batch_axis == axes(ds_drotation, 3) == axes(ds_dtranslation, 2) == axes(ds_dbackground, 1) == axes(ds_dout_weight, 1)
    @argcheck N_out == N_out_p1 - 1

    n_points = length(points)
    @argcheck length(ds_dpoint_weight) == n_points
    batch_size = length(batch_axis)

    ds_dbackground = vec(sum!(reshape(ds_dbackground, ntuple(_ -> 1, Val(N_out))..., batch_size), ds_dout))

    scale = SVector{N_out, T}(size(ds_dout)[1:end-1]) / T(2)
    shifts=DiffPointRasterisation.voxel_shifts(Val(N_out))

    ds_dpoints = fill!(ds_dpoints, zero(TP))
    ds_drotation = fill!(ds_drotation, zero(TR))
    ds_dtranslation = fill!(ds_dtranslation, zero(TT))
    ds_dout_weight = fill!(ds_dout_weight, zero(OW))
    ds_dpoint_weight = fill!(ds_dpoint_weight, zero(PW))

    args = (T, ds_dout, points, rotation, translation, out_weight, point_weight, shifts, scale, ds_dpoints, ds_drotation, ds_dtranslation, ds_dout_weight, ds_dpoint_weight)

    ndrange = (n_points, batch_size, 2^N_out)

    workgroup_size(threads) = (1, min(threads ÷ (2^N_out), batch_size), 2^N_out)

    function shmem(threads)
        _, bs_p_wg, n_voxel =  workgroup_size(threads)
        ((N_out + 1) * n_voxel + N_in) * bs_p_wg * sizeof(T)
        # ((N_out + 1) * threads + N_in * bs_p_wg) * sizeof(T)
    end

    let kernel = @cuda launch=false raster_pullback_kernel!(args...)
        config = CUDA.launch_configuration(kernel.fun; shmem)
        workgroup_sz = workgroup_size(config.threads)
        blocks = cld.(ndrange, workgroup_sz)

        kernel(args...; threads=workgroup_sz, blocks=blocks, shmem=shmem(config.threads))
    end

    return (;
        points=ds_dpoints,
        rotation=ds_drotation,
        translation=ds_dtranslation,
        background=ds_dbackground,
        out_weight=ds_dout_weight,
        point_weight=ds_dpoint_weight,
    )
end


DiffPointRasterisation.default_ds_dpoints_batched(points::CuVector{<:AbstractVector{TP}}, N_in, batch_size) where {TP<:Number} = similar(points, TP, (N_in, length(points)))

DiffPointRasterisation.default_ds_dpoint_weight_batched(points::CuVector{<:AbstractVector{<:Number}}, T, batch_size) = similar(points, T)

end  # module