module DiffPointRasterisationCUDAExt

using DiffPointRasterisation, CUDA

function raster!(
    out::AbstractArray{T, N_out},
    points::AbstractMatrix{T},
    rotation::StaticMatrix{N_in, N_in, T},
    translation::AbstractVector{T},
    background::T=zero(T),
    weight::T=one(T)/size(points, 2),
) where {N_in, N_out, T}
    fill!(out, background)
    origin = (-@SVector ones(T, N_out)) - translation
    projection_idxs = SVector(ntuple(identity, N_out))
    scale = SVector{N_out, T}(size(out)) / 2 
    half = T(0.5)
    shifts=voxel_shifts(Val(N_out))

    all_density_idxs = CartesianIndices(out)

    for point in eachcol(points)
        # coordinate of transformed point in output coordinate system
        # which is defined by the (integer) coordinates of the pixels/voxels
        # in the output array. 
        coord = ((rotation * point)[projection_idxs] - origin) .* scale
        # round to lower integer coordinate ("lower left" neighboring pixel)
        idx_lower = round.(Int, coord .- half, RoundUp)
        # distance to lower integer coordinate (distance from "lower left" neighboring pixel)
        deltas_lower = coord - (idx_lower .- half)
        # distances to lower (first column) and upper (second column) integer coordinates
        deltas = [deltas_lower 1 .- deltas_lower]

        @inbounds for shift in shifts  # loop over neighboring pixels/voxels
            # index of neighboring pixel/voxel
            voxel_idx = CartesianIndex(idx_lower.data .+ shift)
            (voxel_idx in all_density_idxs) || continue
            val = one(T)
            for i in 1:N_out
                val *= deltas[i, mod1(shift[i], 2)]  # product of distances along each coordinate axis
            end
            out[voxel_idx] += val * weight  # fill neighboring pixel/voxel
        end
    end
    out
end


function raster!(
    out::CuArray{T, N},
    points::CuMatrix{T},
    rotation::StaticMatrix{N_in, N_in, T},
    translation::CuVector{T},
    background::T=zero(T),
    weight::T=one(T)/size(points, 2),
) where {N, T<:Real, TT}
    @assert size(points, 1) == N

    fill!(out, zero(TT))
    scale = one(T) / grid.spacing

    args = (out, points, grid.origin, scale, norm_factor, shifts, grid.size)
    kernel = @cuda launch=false raster_kernel!(args...)
    kernel_config = launch_configuration(kernel.fun)
    threads = (cld(kernel_config.threads, 2^N), 2^N)
    blocks = (cld(size(points, 2), threads[1]), 1)

    kernel(args...; threads, blocks)
    out
end

function raster_kernel!(density::DenseArray{TT, N}, points::DenseMatrix{T}, origin::DenseVector{T}, scale::T, norm_factor::T, shifts, grid_size) where {N, T<:Real, TT}
    point_offset = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    point_stride = blockDim().x * gridDim().x 
    voxel = threadIdx().y

    for point_id in point_offset:point_stride:size(points, 2)
        val = norm_factor
        voxel_idx = 0
        stride = 1
        for dim in 1:N
            point_coord_idx = (point_id - 1) * N + dim
            coord = (points[point_coord_idx] - origin[dim]) * scale
            ilf = round(coord - T(0.5), RoundUp)
            shift = shifts[voxel][dim]
            delta = coord - (ilf - T(0.5))
            if shift == 0
                delta = one(T) - delta
            end
            val *= delta

            idx_dim_lower = Int(ilf)
            idx_dim = idx_dim_lower + shift
            if idx_dim > grid_size[dim] || idx_dim < 1
                # Out of grid.
                # Set voxel_idx to a value that guarantees it won't 
                # be added to density (see below) and break.
                voxel_idx = length(density) + 1
                break
            end
            voxel_idx += (dim == 1 ? idx_dim : idx_dim - 1) * stride
            stride *= grid_size[dim]
        end
        if voxel_idx <= length(density)
            # CUDA.atomic_add!(CUDA.pointer(density, voxel_idx), val)
            CUDA.@atomic density[voxel_idx] += val
        end
    end
    nothing
end

end  # module DiffPointRasterisationCUDAExt