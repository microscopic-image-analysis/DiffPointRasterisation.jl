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
using StaticArrays


function raster_pullback_kernel!(
    ::Val{N_in},
    ds_dout::AbstractArray{T, N_out_p1},
    points,
    rotations,
    translations,
    weights,
    shifts,
    scale,
    projection_idxs,
    # outputs:
    ds_dpoints,
    ds_dprojection_rotation,
    ds_dtranslation,
    ds_dweight,

) where {T, N_in, N_out_p1}
    N_out = N_out_p1 - 1  # dimensionality of output, without batch dimension

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
    in_batch = batch_idx <= size(rotations, 3)

    dimension = (N_out, n_voxel, batchsize_per_workgroup)
    ds_dpoint_rot = CuDynamicSharedArray(T, dimension)
    ds_dpoint_local = CuDynamicSharedArray(T, (N_in, batchsize_per_workgroup), sizeof(T) * prod(dimension))

    rotation = in_batch ? @inbounds(SMatrix{N_in, N_in, T}(@view rotations[:, :, batch_idx])) : @SMatrix zeros(T, N_in, N_in)
    point = @inbounds SVector{N_in, T}(@view points[:, point_idx])

    if in_batch
        translation = @inbounds SVector{N_out, T}(@view translations[:, batch_idx])
        weight = @inbounds weights[batch_idx]
        shift = @inbounds shifts[neighbor_voxel_id]
        origin = (-@SVector ones(T, N_out)) - translation

        coord_reference_voxel, deltas = DiffPointRasterisation.reference_coordinate_and_deltas(
            point,
            rotation,
            projection_idxs,
            origin,
            scale,
        )
        voxel_idx = CartesianIndex(CartesianIndex(Tuple(coord_reference_voxel)) + CartesianIndex(shift), batch_idx)


        ds_dweight_local = zero(T)
        if voxel_idx in CartesianIndices(ds_dout)
            @inbounds ds_dweight_local = DiffPointRasterisation.voxel_weight(
                deltas,
                shift,
                projection_idxs,
                ds_dout[voxel_idx],
            )

            factor = ds_dout[voxel_idx] * weight
            ds_dcoord_part = SVector(factor .* ntuple(n -> DiffPointRasterisation.interpolation_weight(n, N_out, deltas, shift), N_out))
            @inbounds ds_dpoint_rot[:, s, b] .= ds_dcoord_part .* scale
        else
            @inbounds ds_dpoint_rot[:, s, b] .= zero(T)
        end

        @inbounds CUDA.@atomic ds_dweight[batch_idx] += ds_dweight_local
    else
        @inbounds ds_dpoint_rot[:, s, b] .= zero(T)
    end

    # parallel summation of ds_dpoint_rot over neighboring-voxel dimension
    # for a given thread-local batch index
    stride = 1
    @inbounds while stride < n_voxel
        sync_threads()
        idx = 2 * stride * (s - 1) + 1
        dim = 1
        while dim <= N_out
            if idx <= n_voxel
                other_val = if idx + stride <= n_voxel
                    ds_dpoint_rot[dim, idx + stride, b]
                else
                    zero(T)
                end
                ds_dpoint_rot[dim, idx, b] += other_val
            end
            dim += 1
        end
        stride *= 2
    end

    sync_threads()

    if in_batch
        dim = s
        if dim <= N_out
            coef = ds_dpoint_rot[dim, 1, b]
            @inbounds CUDA.@atomic ds_dtranslation[dim, batch_idx] += coef
            j = 1
            while j <= N_in
                val = coef * point[j]
                @inbounds CUDA.@atomic ds_dprojection_rotation[dim, j, batch_idx] += val 
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
            @inbounds val += rotation[j, dim] * ds_dpoint_rot[j, 1, b]
            j += 1
        end
        @inbounds ds_dpoint_local[dim, b] = val
        dim += n_voxel
    end

    # parallel summation of ds_dpoint_local over batch dimension
    stride = 1
    @inbounds while stride < batchsize_per_workgroup
        sync_threads()
        idx = 2 * stride * (b - 1) + 1
        dim = s 
        while dim <= N_in
            if idx <= batchsize_per_workgroup
                other_val = if idx + stride <= batchsize_per_workgroup
                    ds_dpoint_local[dim, idx + stride]
                else
                    zero(T)
                end
                ds_dpoint_local[dim, idx] += other_val
            end
            dim += n_voxel
        end
        stride *= 2
    end

    sync_threads()

    dim = thread
    while dim <= N_in
        val = ds_dpoint_local[dim, 1]
        # batch might be split across blocks, so need atomic add
        @inbounds CUDA.@atomic ds_dpoints[dim, point_idx] += val
        dim += n_threads_per_workgroup
    end

    nothing
end


function DiffPointRasterisation.raster_pullback!(
    ::Val{N_in},
    ds_dout::CuArray{T, N_out_p1},
    points::CuMatrix{<:Number},
    rotation::CuArray{<:Number, 3},
    translation::CuMatrix{<:Number},
    # TODO: for some reason type inference fails if the following
    # two arrays are FillArrays... 
    background::CuVector{<:Number}=CUDA.zeros(T, size(rotation, 3)),
    weight::CuVector{<:Number}=CUDA.ones(T, size(rotation, 3)),
) where {T<:Number, N_in, N_out_p1}
    N_out = N_out_p1 - 1
    out_batch_dim = ndims(ds_dout)
    batch_axis = axes(ds_dout, out_batch_dim)
    n_points = size(points, 2)
    batch_size = length(batch_axis)
    @argcheck axes(ds_dout, out_batch_dim) == axes(rotation, 3) == axes(translation, 2) == axes(background, 1) == axes(weight, 1)

    ds_dbackground = dropdims(sum(ds_dout; dims=1:N_out); dims=ntuple(identity, Val(N_out)))

    scale = SVector{N_out, T}(size(ds_dout)[1:end-1]) / T(2)
    projection_idxs = SVector{N_out}(ntuple(identity, N_out))
    shifts=DiffPointRasterisation.voxel_shifts(Val(N_out))

    ds_dpoints = fill!(similar(points), zero(T))
    ds_drotation = fill!(similar(rotation), zero(T))
    ds_dtranslation = fill!(similar(translation), zero(T))
    ds_dweight = fill!(similar(weight), zero(T))

    args = (Val(N_in), ds_dout, points, rotation, translation, weight, shifts, scale, projection_idxs, ds_dpoints, ds_drotation, ds_dtranslation, ds_dweight)

    ndrange = (n_points, batch_size, 2^N_out)

    workgroup_size(threads) = (1, min(threads ÷ (2^N_out), batch_size), 2^N_out)

    function shmem(threads)
        batchsize_per_workgroup =  workgroup_size(threads)[2]
        (N_out * threads + N_in * batchsize_per_workgroup) * sizeof(T)
    end

    let kernel = @cuda launch=false raster_pullback_kernel!(args...)
        config = CUDA.launch_configuration(kernel.fun; shmem)
        workgroup_sz = workgroup_size(config.threads)
        blocks = cld.(ndrange, workgroup_sz)
        kernel(args...; threads=workgroup_sz, blocks=blocks, shmem=shmem(prod(workgroup_sz)))
    end

    return (;
        points=ds_dpoints,
        rotation=ds_drotation,
        translation=ds_dtranslation,
        background=ds_dbackground,
        weight=ds_dweight,
    )
end

end  # module