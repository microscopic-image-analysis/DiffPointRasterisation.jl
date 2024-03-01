
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

canonical_arg(arg::Number) = @SVector [arg]

canonical_arg(arg::AbstractVector{<:Number}) = arg

canonical_arg(arg::AbstractVector{<:StaticArray}) = arg

canonical_arg(arg::AbstractVector{<:AbstractArray{<:Number}}) = canonical_arg(arg, Val(size(arg[1])))

canonical_arg(arg::AbstractVector{<:AbstractArray{T}}, ::Val{sz}) where {sz, T<:Number} = SizedArray{Tuple{sz...}, T}.(arg)

@testitem "canonical_arg" begin
    using StaticArrays
    @testset "number" begin
        inp = randn()
        @inferred DiffPointRasterisation.canonical_arg(inp)
        out = DiffPointRasterisation.canonical_arg(inp)
        @test out[] == inp
        @test out isa StaticVector{1}
    end
    @testset "vector" begin
        inp = randn(3)
        @inferred DiffPointRasterisation.canonical_arg(inp)
        out = DiffPointRasterisation.canonical_arg(inp)
        @test out === inp
    end

    @testset "vec of dynamic vec" begin
        inp = [randn(3) for _ in 1:5]
        out = DiffPointRasterisation.canonical_arg(inp)
        @test out == inp
        @test out isa Vector{<:StaticVector{3}}
    end

    @testset "vec of static vec" begin
        inp = [@SVector randn(3) for _ in 1:5]
        @inferred DiffPointRasterisation.canonical_arg(inp)
        out = DiffPointRasterisation.canonical_arg(inp)
        @test out === inp
        @test out isa Vector{<:StaticVector{3}}
    end

    @testset "vec of dynamic matrix" begin
        inp = [randn(3, 2) for _ in 1:5]
        out = DiffPointRasterisation.canonical_arg(inp)
        @test out == inp
        @test out isa Vector{<:StaticMatrix{3, 2}}
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