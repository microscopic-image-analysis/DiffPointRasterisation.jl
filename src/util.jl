
"""
    digitstuple(k, Val(N))

Return a N-tuple containing the bit-representation of k
"""
digitstuple(k, ::Val{N}, int_type=Int64) where {N} = ntuple(i -> int_type(k>>(i-1) % 2), N)

@testitem "digitstuple" begin
    @test DiffPointRasterisation.digitstuple(5, Val(3)) == (1, 0, 1) 
    @test DiffPointRasterisation.digitstuple(2, Val(2)) == (0, 1) 
    @test DiffPointRasterisation.digitstuple(2, Val(4)) == (0, 1, 0, 0) 
end

"""
    voxel_shifts(Val(N), [int_type])

Enumerate nearest neighbor coordinate shifts with respect
to "upper left" voxel.

For a N-dimensional voxel grid, return a 2^N-tuple of N-tuples,
where each element of the outer tuple is a cartesian coordinate
shift from the "upper left" voxel.
"""
voxel_shifts(::Val{N}, int_type=Int64) where {N} = ntuple(k -> digitstuple(k-1, Val(N), int_type), 2^N)

canonical_arg(arg::Number) = SVector{1}(arg)

canonical_arg(arg::AbstractVector{<:Number}) = arg

canonical_arg(arg::AbstractVector{<:StaticArray}) = arg

canonical_arg(arg::AbstractVector{<:AbstractArray}) = SizedArray{Tuple{size(arg[1])...}}.(arg)

canonical_arg(arg::AbstractMatrix{T}) where {T<:Number} = reinterpret(reshape, SVector{size(arg, 1), T}, arg)

function canonical_arg(arg::AbstractArray{T, 3}) where {T<:Number}
    N = size(arg, 1)
    M = size(arg, 2)
    L = N*M
    reinterpret(reshape, SMatrix{N, M, T, L}, reshape(arg, L, :))
end

@testitem "canonical_arg" begin
    @testset "vector" begin
        v = randn(3)
        @test DiffPointRasterisation.canonical_arg(v) == v
    end

    @testset "matrix" begin
        v_of_v = [randn(3) for _ in 1:5]
        m = stack(v_of_v)
        @test DiffPointRasterisation.canonical_arg(m) == v_of_v
    end

    @testset "3d array" begin
        v_of_m = [randn(3, 2) for _ in 1:5]
        a = stack(v_of_m)
        @test DiffPointRasterisation.canonical_arg(a) == v_of_m
    end
end


@inline append_singleton_dim(a) = reshape(a, size(a)..., 1)

@inline append_singleton_dim(a::Number) = [a]

@inline drop_last_dim(a) = dropdims(a; dims=ndims(a))

@testitem "append drop dim" begin
    using BenchmarkTools
    a = randn(2, 3, 4)
    a2 = DiffPointRasterisation.drop_last_dim(DiffPointRasterisation.append_singleton_dim(a))
    @test a2 === a broken=true

    allocations = @ballocated DiffPointRasterisation.drop_last_dim(DiffPointRasterisation.append_singleton_dim($a)) evals=1 samples=1
    @test allocations == 0 broken=true
end