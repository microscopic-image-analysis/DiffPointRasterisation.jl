module DiffPointRasterisationCUDAExt

using DiffPointRasterisation, CUDA


function raster_pullback_kernel!(
    ::Val{N_in},
    ds_dout::CuArray{T, N_out_p1},
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

    n_voxel = blockDim().x
    points_per_workgroup = blockDim().y
    batchsize_per_workgroup = blockDim().z
    @assert points_per_workgroup == 1
    @assert n_voxel == 2^N_out
    n_threads_per_workgroup = n_voxel * batchsize_per_workgroup

    s = threadIdx().x
    b = threadIdx().z
    thread = (b - 1) * n_voxel + s

    neighbor_voxel_id = (blockIdx().x - 1) * n_voxel + s
    point_idx = (blockIdx().y - 1) * points_per_workgroup + threadIdx().y
    batch_idx = (blockIdx().z - 1) * batchsize_per_workgroup + b

    ds_dpoint_rot = CuStaticSharedArray(T, (N_out, n_voxel, batchsize_per_workgroup))
    ds_dweight_local = zero(T)

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


    if voxel_idx in CartesianIndices(ds_dout)
        @inbounds ds_dweight_local = voxel_weight(
            deltas,
            shift,
            projection_idxs,
            ds_dout[voxel_idx],
        )

        factor = ds_dout[voxel_idx] * weight
        ds_dcoord_part = SVector(factor .* ntuple(n -> interpolation_weight(n, N_out, deltas, shift), N_out))
        @inbounds ds_dpoint_rot[:, s, b] = ds_dcoord_part .* scale
    else
        @inbounds ds_dpoint_rot[:, s, b] .= zero(T)
    end

    @inbounds CUDA.@atomic ds_dweight[batch_idx] += ds_dweight_local

    # parallel summation of ds_dpoint_rot over neighboring-voxel dimension
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
    i = thread
    while i <= N_out
        val = ds_dpoint_rot[i, 1, b]
        @inbounds CUDA.@atomic ds_dtranslation[i, batch_idx] += val
        coef = ds_dpoint_rot[i, 1, b]
        j = 1
        while j <= N_in
            val = coef * point[j]
            @inbounds CUDA.@atomic ds_dprojection_rotation[i, j, batch_idx] += val 
            j += 1
        end
        i += n_threads_per_workgroup
    end

    # parallel summation of ds_dpoint_rot over batch dimension
    stride = 1
    @inbounds while stride < batchsize_per_workgroup
        sync_threads()
        idx = 2 * stride * (b - 1) + 1
        dim = 1
        while dim <= N_out
            if idx <= batchsize_per_workgroup
                other_val = if idx + stride <= batchsize_per_workgroup
                    ds_dpoint_rot[dim, 1, idx + stride]
                else
                    zero(T)
                end
                ds_dpoint_rot[dim, 1, idx] += other_val
            end
            dim += 1
        end
        stride *= 2
    end

    sync_threads()
    i = thread
    while i <= N_in
        val = zero(T)
        j = 1
        while j <= N_out
            val += rotation[j, i] * ds_dpoint_rot[j, 1, 1]
            j += 1
        end
        @inbounds ds_dpoints[i, point_idx] = val
        i += n_threads_per_workgroup
    end

    nothing
end


function DiffPointRasterisation._raster_pullback!(
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

    scale = SVector{N_out, T}(size(ds_dout)[1:end-1]) / T(2)
    projection_idxs = SVector{N_out}(ntuple(identity, N_out))
    shifts=voxel_shifts(Val(N_out))

    ds_dpoints = similar(points)
    ds_dprojection_rotation = similar(rotation, (N_out, N_in, batch_size))
    ds_dtranslation = similar(translation)
    ds_dweight = similar(weight)

    args = (Val(N_in), ds_dout, points, rotation, translation, weight, shifts, scale, projection_idxs, ds_dpoints, ds_dprojection_rotation, ds_dtranslation, ds_dweight)

    ndrange = (2^N_out, n_points, batch_size)

    let kernel = @cuda launch=false raster_pullback_kernel!(args...)
        config = CUDA.launch_configuration(kernel.fun)
        workgroup_size = (2^N_out, 1, config.threads ÷ (2^N_out))
        kernel(args...; threads=workgroup_size, blocks=cld.(ndrange, workgroup_size))
    end


    ds_drotation = N_out == N_in ? ds_dprojection_rotation : vcat(ds_dprojection_rotation, KernelAbstractions.zeros(backend, T, 1, N_in, batch_size))
    ds_dbackground = dropdims(sum(ds_dout; dims=1:N_out); dims=1:N_out)

    return (;
        points=ds_dpoints,
        rotation=ds_drotation,
        translation=ds_dtranslation,
        background=ds_dbackground,
        weight=ds_dweight,
    )
end

end  # module