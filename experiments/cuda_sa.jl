using CUDA
using StaticArrays


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