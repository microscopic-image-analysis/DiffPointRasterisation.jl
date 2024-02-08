using CUDA
using DiffPointRasterisation
using Rotations
using StaticArrays

CUDA.allowscalar(false)

function rotate_batch_as_vec(out, points, rotations)
    batch_size = length(out)
    @assert batch_size == length(points) == length(rotations)

    function rotate_kernel(out, points, rotations)
        coord_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        point_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        batch_idx = (blockIdx().z - 1) * blockDim().z + threadIdx().z
        if batch_idx <= length(rotations)
            out_batch = out[batch_idx]
            points_batch = points[batch_idx]
            N = size(points_batch, 1)
            rotation = rotations[batch_idx]
            if point_idx <= size(points_batch, 2)
                point = @view points_batch[:, point_idx]
                point_out = @view out_batch[:, point_idx]
                if coord_idx <= N
                    val = zero(eltype(out_batch))
                    for i in 1:N
                        val += rotation[coord_idx, i] * point[i]
                    end
                    point_out[coord_idx] = val
                end
            end
        end
        nothing
    end

    dim_points = size(points[1], 1)
    n_points = size(points[1], 2)

    args = (out, points, rotations)
    let kernel = @cuda launch=false rotate_kernel(args...)
        kernel_config = launch_configuration(kernel.fun)
        available_threads = kernel_config.threads
        tx = min(dim_points, available_threads)
        available_threads = cld(available_threads, tx)
        ty = min(n_points, available_threads)
        available_threads = cld(available_threads, ty)
        tz = min(batch_size, available_threads)

        threads = (tx, ty, tz)
        blocks = cld.(threads, (dim_points, n_points, batch_size))

        kernel(args...; threads, blocks)
    end

    out
end


function do_raster(;batch_size=3)
    grid_size = (8, 8, 8)
    points = 0.5f0 .* CUDA.randn(3, 1000)
    rotation = cu(stack(rand(RotMatrix3{Float32}, batch_size)))
    translation = CUDA.zeros(3, batch_size)
    background = cu(Float32.(collect(1:batch_size)))
    weight = 10 .* CUDA.ones(batch_size)

    out = raster(grid_size, points, rotation, translation, background, weight)
end


function staticarrays()
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