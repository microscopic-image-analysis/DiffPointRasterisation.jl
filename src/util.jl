
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


prefix(s::Symbol) = Symbol("ds_d" * string(s))


@inline append_singleton_dim(a) = reshape(a, size(a)..., 1)

@inline drop_last_dim(a) = dropdims(a; dims=ndims(a))

@testitem "append drop dim" begin
    using BenchmarkTools
    a = randn(2, 3, 4)
    a2 = DiffPointRasterisation.drop_last_dim(DiffPointRasterisation.append_singleton_dim(a))
    @test a2 === a broken=true

    allocations = @ballocated DiffPointRasterisation.drop_last_dim(DiffPointRasterisation.append_singleton_dim($a)) evals=1 samples=1
    @test allocations == 0 broken=true
end