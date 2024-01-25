"""
    raster(grid_size, points, rotation, translation, [background, weight])

Interpolate points (multi-) linearly into an Nd-array of size `grid_size`.

Before `points` are interpolated into the array, each point ``p`` is first
transformed according to
```math
\\hat{p} = R p + t
```
with `rotation` ``R`` and `translation` ``t``.

Points ``\\hat{p}`` that fall into the N-dimensional hypercube
with edges spanning from (-1, 1) in each dimension, are interpolated
into the output array.

The total `weight` of each point is distributed onto the 2^N nearest
pixels/voxels of the output array (according to the closeness of the
voxel center to the coordinates of point ``\\hat{p}``) via
N-linear interpolation.
"""
raster(
    grid_size,
    points::AbstractMatrix{T},
    rotation,
    translation,
    background=isa(rotation, AbstractMatrix{<:Number}) ? zero(T) : Zeros(T, length(rotation)),
    weight=isa(rotation, AbstractMatrix{<:Number}) ? one(T) : Ones(T, length(rotation)),
) where {T} = raster!(
    isa(rotation, AbstractMatrix{<:Number}) ? similar(points, grid_size) : [similar(points, grid_size) for _ in 1:length(rotation)],
    points,
    rotation,
    translation,
    background,
    weight,
)

@testitem "raster single point 2D" begin
    using Rotations, StaticArrays
    grid_size = (5, 5)

    points_single_center = zeros(2, 1) 
    no_rotation = SizedMatrix{2, 2}([1.0;0.0;;0.0;1.0;;])
    no_translation = zeros(2)
    zero_background = 0.0
    unit_weight = 1.0

    out = raster(grid_size, points_single_center, no_rotation, no_translation, zero_background, unit_weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 1 0 0
        0 0 0 0 0
        0 0 0 0 0
    ]
    
    points_single_1pix_right = [0.0;0.4;;]
    out = raster(grid_size, points_single_1pix_right, no_rotation, no_translation, zero_background, unit_weight)
    @test out ≈ [
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 1 0
        0 0 0 0 0
        0 0 0 0 0
    ]

    points_single_halfpix_down = [0.2;0.0;;]
    out = raster(grid_size, points_single_halfpix_down, no_rotation, no_translation, zero_background, unit_weight)
    @test out ≈ [
        0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.5 0.0 0.0
        0.0 0.0 0.5 0.0 0.0
        0.0 0.0 0.0 0.0 0.0
    ]

    points_single_halfpix_down_and_right = [0.2;0.2;;]
    out = raster(grid_size, points_single_halfpix_down_and_right, no_rotation, no_translation, zero_background, unit_weight)
    @test out ≈ [
        0.00 0.00 0.00 0.00 0.00
        0.00 0.00 0.00 0.00 0.00
        0.00 0.00 0.25 0.25 0.00
        0.00 0.00 0.25 0.25 0.00
        0.00 0.00 0.00 0.00 0.00
    ]
end


"""
    raster_project(grid_size, points, rotation, translation, [background, weight])

Interpolate N-dimensional points (multi-) linearly into an N-1 dimensional-array
of size `grid_size`.

Before `points` are interpolated into the array, each point ``p`` is first
transformed according to
```math
\\hat{p} = P R p + t
```
with N-dimensional `rotation` ``R``, projection ``P`` and N-1-dmensional
`translation` ``t``.
The projection simply drops the last coordinate of ``R p``.

Points ``\\hat{p}`` that fall into the N-1-dimensional hypercube
with edges spanning from (-1, 1) in each dimension, are interpolated
into the output array.

The total `weight` of each point is distributed onto the 2^(N-1) nearest
pixels/voxels of the output array (according to the closeness of the
voxel center to the coordinates of point ``\\hat{p}``) via
N-1-linear interpolation.
"""
raster_project(
    grid_size,
    points::AbstractMatrix{T},
    rotation,
    translation,
    background=isa(rotation, AbstractMatrix{<:Number}) ? zero(T) : Zeros(T, length(rotation)),
    weight=isa(rotation, AbstractMatrix{<:Number}) ? one(T) : Ones(T, length(rotation)),
) where {T} = raster_project!(
    isa(rotation, AbstractMatrix{<:Number}) ? similar(points, grid_size) : [similar(points, grid_size) for _ in 1:length(rotation)],
    points,
    rotation,
    translation,
    background,
    weight,
)


"""
    raster!(out, points, rotation, translation, [background, weight])

Inplace version of `raster`.

Write output into `out` and return `out`.
"""
raster!(
    out::Union{AbstractArray{T, N_out}, AbstractVector{<:AbstractArray{T, N_out}}},
    points,
    rotation,
    translation,
    background=isa(out, AbstractArray{T}) ? zero(T) : Zeros(T, length(out)),
    weight=isa(out, AbstractArray{T}) ? one(T) : Ones(T, length(out)),
) where {N_out, T<:Number} = _raster!(
    Val(N_out),
    out,
    points,
    rotation,
    translation,
    background,
    weight,
)

@testitem "raster! allocations" begin
    using BenchmarkTools, Rotations
    out = zeros(8, 8, 8)
    points = randn(3, 10)
    rotation = rand(QuatRotation)
    translation = zeros(3)

    allocations = @ballocated DiffPointRasterisation.raster!($out, $points, $rotation, $translation) evals=1 samples=1
    @test allocations == 0
end


"""
    raster_project!(out, points, rotation, translation, [background, weight])

Inplace version of `raster_project`.

Write output into `out` and return `out`.
"""
raster_project!(
    out::Union{AbstractArray{T, N_out}, AbstractVector{<:AbstractArray{T, N_out}}},
    points,
    rotation,
    translation,
    background=isa(out, AbstractArray{T}) ? zero(T) : Zeros(T, length(out)),
    weight=isa(out, AbstractArray{T}) ? one(T) : Ones(T, length(out)),
) where {N_out, T<:Number} = _raster!(
    Val(N_out + 1),
    out,
    points,
    rotation,
    translation,
    background,
    weight,
)

@testitem "raster_project! allocations" begin
    using BenchmarkTools, Rotations
    out = zeros(16, 16)
    points = randn(3, 10)
    rotation = rand(QuatRotation)
    translation = zeros(2)

    allocations = @ballocated DiffPointRasterisation.raster_project!($out, $points, $rotation, $translation) evals=1 samples=1
    @test allocations == 0
end


function _raster!(
    ::Val{N_in},
    out::AbstractArray{T, N_out},
    points::AbstractMatrix{<:Number},
    rotation::AbstractMatrix{<:Number},
    translation::AbstractVector{<:Number},
    background::Number,
    weight::Number,
) where {N_in, N_out, T}
    @assert size(points, 1) == size(rotation, 1) == size(rotation, 2) == N_in
    @assert length(translation) == N_out

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

function _raster!(
    ::Val{N_in},
    out::AbstractVector{<:AbstractArray},
    points::AbstractMatrix{<:Number},
    rotation::AbstractVector{<:AbstractMatrix{<:Number}},
    translation::AbstractVector{<:AbstractVector{<:Number}},
    background::AbstractVector{<:Number},
    weight::AbstractVector{<:Number},
) where {N_in}
    Threads.@threads for (idxs, ichunk) in chunks(eachindex(out, rotation, translation, background, weight), Threads.nthreads())
        for i in idxs
            _raster!(Val(N_in), out[i], points, rotation[i], translation[i], background[i], weight[i])
        end
    end
    out
end

@testitem "raster! threaded" begin
    using BenchmarkTools, Rotations
    include("testing.jl")

    batch_size = batch_size_for_test()

    out = [zeros(8, 8, 8) for _ in 1:batch_size]
    out_threaded = [zeros(8, 8, 8) for _ in 1:batch_size]
    points = 0.3 .* randn(3, 10)
    rotation = [rand(QuatRotation) for _ in 1:batch_size]
    translation = [zeros(3) for _ in 1:batch_size]
    background = zeros(batch_size)
    weight = ones(batch_size)

    for i in 1:batch_size
        raster!(out[i], points, rotation[i], translation[i], background[i], weight[i])
    end

    DiffPointRasterisation.raster!(out_threaded, points, rotation, translation, background, weight)

    @test out_threaded ≈ out
end

@testitem "raster_project! threaded" begin
    using BenchmarkTools, Rotations
    include("testing.jl")

    batch_size = batch_size_for_test()

    out = [zeros(16, 16) for _ in 1:batch_size]
    out_threaded = [zeros(16, 16) for _ in 1:batch_size]
    points = 0.3 .* randn(3, 10)
    rotation = [rand(QuatRotation) for _ in 1:batch_size]
    translation = [zeros(2) for _ in 1:batch_size]
    background = zeros(batch_size)
    weight = ones(batch_size)

    for i in 1:batch_size
        DiffPointRasterisation.raster_project!(out[i], points, rotation[i], translation[i], background[i], weight[i])
    end

    DiffPointRasterisation.raster_project!(out_threaded, points, rotation, translation, background, weight)

    @test out_threaded ≈ out
end


"""
    raster_pullback!(
        ds_dout, points, rotation, translation, [background, weight];
        [ds_dpoints, ds_drotation, ds_dtranslation, ds_dbackground, ds_dweight]
    )

Pullback for `raster(...)`/`raster!(...)`.

Take as input `ds_dout` the sensitivity of some quantity (`s` for "scalar")
to the *output* `out` of the function `raster(args...)`, as well as
the exact same arguments `args` that were passed to `raster`, and
return the sensitivities of `s` to the *inputs* `args` of the function
`raster()`/`raster!()`.

Optionally, pre-allocated output arrays for each input sensitivity can be
specified as `ds_d\$INPUT_NAME`, e.g. `ds_dtranslation = [zeros(2) for _ in 1:8]`
for 2-dimensional points and a batch size of 8.
"""
raster_pullback!(
    ds_dout::Union{AbstractArray{<:Number, N_out}, AbstractVector{<:AbstractArray{<:Number, N_out}}},
    args...;
    prealloc...
) where {N_out} = _raster_pullback!(
    Val(N_out),
    ds_dout,
    args...;
    prealloc...
)



"""
    raster_project_pullback!(
        ds_dout, points, rotation, translation, [background, weight];
        [ds_dpoints, ds_drotation, ds_dtranslation, ds_dbackground, ds_dweight]
    )

Pullback for `raster_project(...)`/`raster_project!(...)`.

Take as input `ds_dout` the sensitivity of some quantity (`s` for "scalar")
to the *output* `out` of the function `raster_project(args...)`, as well as
the exact same arguments `args` that were passed to `raster_project`, and
return the sensitivities of `s` to the *inputs* `args` of the function
`raster_project()`/`raster_project!()`.

Optionally, pre-allocated output arrays for each input sensitivity can be
specified as `ds_d\$INPUT_NAME`, e.g. `ds_dtranslation = [zeros(2) for _ in 1:8]`
for 3-dimensional points and a batch size of 8.
"""
raster_project_pullback!(
    ds_dout::Union{AbstractArray{<:Number, N_out}, AbstractVector{<:AbstractArray{<:Number, N_out}}},
    args...;
    prealloc...
) where {N_out} = _raster_pullback!(
    Val(N_out + 1),
    ds_dout,
    args...;
    prealloc...
)


function _raster_pullback!(
    ::Val{N_in},
    ds_dout::AbstractArray{T, N_out},
    points::AbstractMatrix{<:Number},
    rotation::AbstractMatrix{<:Number},
    translation::AbstractVector{<:Number},
    background::Number=zero(T),
    weight::Number=one(T);
    accumulate_prealloc=false,
    prealloc...,
) where {N_in, N_out, T}
    # The strategy followed here is to redo some of the calculations
    # made in the forward pass instead of caching them in the forward
    # pass and reusing them here.
    args = (;points,)
    @unpack ds_dpoints = _pullback_alloc_serial(args, NamedTuple(prealloc))
    accumulate_prealloc || fill!(ds_dpoints, zero(T))

    origin = (-@SVector ones(T, N_out)) - translation
    projection_idxs = SVector(ntuple(identity, N_out))
    scale = SVector{N_out, T}(size(ds_dout)) / 2 
    half = T(0.5)
    shifts=voxel_shifts(Val(N_out))
    all_density_idxs = CartesianIndices(ds_dout)

    # initialize some output for accumulation
    ds_dtranslation = @SVector zeros(T, N_out)
    ds_dprojection_rotation = @SMatrix zeros(T, N_out, N_in)
    ds_dweight = zero(T)

    # loop over points
    for (pt_idx, point) in enumerate(eachcol(points))
        point = SVector{N_in, T}(point)
        coord::SVector{N_out, T} = ((rotation * point)[projection_idxs] - origin) .* scale
        idx_lower = round.(Int, coord .- half, RoundUp)
        deltas_lower = coord - (idx_lower .- half)
        deltas = [deltas_lower 1 .- deltas_lower]

        ds_dcoord = @SVector zeros(T, N_out)
        # loop over voxels that are affected by point
        for shift in shifts
            voxel_idx = CartesianIndex(idx_lower.data .+ shift)
            (voxel_idx in all_density_idxs) || continue

            val = one(T)
            for i in 1:N_out
                val *= deltas[i, mod1(shift[i], 2)]  # product of distances along each coordinate axis
            end
            ds_dweight += ds_dout[voxel_idx] * val

            factor = ds_dout[voxel_idx] * weight
            # loop over dimensions of point
            ds_dcoord += SVector(factor .* ntuple(n -> interpolation_weight(n, N_out, deltas, shift), N_out))
        end
        scaled = ds_dcoord .* scale
        ds_dtranslation += scaled
        ds_dprojection_rotation += scaled * point'
        ds_dpoint = rotation[projection_idxs, :]' * scaled 
        @view(ds_dpoints[:, pt_idx]) .+= ds_dpoint
    end
    ds_drotation = N_out == N_in ? ds_dprojection_rotation : vcat(ds_dprojection_rotation, @SMatrix zeros(T, 1, N_in))
    return (; points=ds_dpoints, rotation=ds_drotation, translation=ds_dtranslation, background=sum(ds_dout), weight=ds_dweight)
end

@testitem "raster_pullback! allocations" begin
    using BenchmarkTools, Rotations
    ds_dout = zeros(8, 8, 8)
    points = randn(3, 10)
    ds_dpoints = similar(points)
    rotation = rand(QuatRotation)
    translation = zeros(3)
    background = 0.0
    weight = 1.0

    allocations = @ballocated DiffPointRasterisation.raster_pullback!(
        $ds_dout,
        $points,
        $rotation,
        $translation,
        $background,
        $weight;
        points=$ds_dpoints,
    ) evals=1 samples=1
    @test allocations == 0
end

function _raster_pullback!(
    ::Val{N_in},
    ds_dout::AbstractVector{<:AbstractArray{T, N_out}},
    points::AbstractMatrix{<:Number},
    rotation::AbstractVector{<:AbstractMatrix{<:Number}},
    translation::AbstractVector{<:AbstractVector{<:Number}},
    # TODO: for some reason type inference fails if the following
    # two arrays are FillArrays... 
    background::AbstractVector{<:Number}=zeros(T, length(rotation)),
    weight::AbstractVector{<:Number}=ones(T, length(rotation));
    prealloc...
) where {N_in, N_out, T}
    args = (;points, rotation, translation, background, weight)
    batch_size = length(translation)
    @unpack ds_dpoints, ds_drotation, ds_dtranslation, ds_dbackground, ds_dweight = _pullback_alloc_threaded(args, NamedTuple(prealloc), min(batch_size, Threads.nthreads()))
    @assert isa(ds_dpoints, AbstractVector{<:AbstractMatrix{<:Number}})

    Threads.@threads for (idxs, ichunk) in chunks(eachindex(ds_dout, rotation, translation, background, weight), length(ds_dpoints))
        fill!(ds_dpoints[ichunk], zero(T))
        for i in idxs
            args_i = (ds_dout[i], points, rotation[i], translation[i], background[i], weight[i])
            result_i = _raster_pullback!(Val(N_in), args_i...; accumulate_prealloc=true, points=ds_dpoints[ichunk])
            ds_drotation[i] .= result_i.rotation
            ds_dtranslation[i] = result_i.translation
            ds_dbackground[i] = result_i.background
            ds_dweight[i] = result_i.weight
        end
    end
    return (; points=sum(ds_dpoints), rotation=ds_drotation, translation=ds_dtranslation, background=ds_dbackground, weight=ds_dweight)
end

@testitem "raster_pullback! threaded" begin
    using BenchmarkTools, Rotations
    include("testing.jl")

    batch_size = batch_size_for_test()

    ds_dout = [randn(8, 8, 8) for _ in 1:batch_size]
    points = 0.3 .* randn(3, 10)
    rotation = [rand(QuatRotation) for _ in 1:batch_size]
    translation = [zeros(3) for _ in 1:batch_size]
    background = zeros(batch_size)
    weight = ones(batch_size)

    ds_dargs_threaded = DiffPointRasterisation.raster_pullback!(ds_dout, points, rotation, translation, background, weight)

    ds_dpoints = Matrix{Float64}[]
    for i in 1:batch_size
        ds_dargs_i = raster_pullback!(ds_dout[i], points, rotation[i], translation[i], background[i], weight[i])
        push!(ds_dpoints, ds_dargs_i.points)
        @test ds_dargs_threaded.rotation[i] ≈ ds_dargs_i.rotation
        @test ds_dargs_threaded.translation[i] ≈ ds_dargs_i.translation
        @test ds_dargs_threaded.background[i] ≈ ds_dargs_i.background
        @test ds_dargs_threaded.weight[i] ≈ ds_dargs_i.weight
    end
    @test ds_dargs_threaded.points ≈ sum(ds_dpoints)
end

@testitem "raster_project_pullback! threaded" begin
    using BenchmarkTools, Rotations
    include("testing.jl")

    batch_size = batch_size_for_test()

    ds_dout = [randn(16, 16) for _ in 1:batch_size]
    points = 0.3 .* randn(3, 10)
    rotation = [rand(QuatRotation) for _ in 1:batch_size]
    translation = [zeros(2) for _ in 1:batch_size]
    background = zeros(batch_size)
    weight = ones(batch_size)

    ds_dargs_threaded = DiffPointRasterisation.raster_project_pullback!(ds_dout, points, rotation, translation, background, weight)

    ds_dpoints = Matrix{Float64}[]
    for i in 1:batch_size
        ds_dargs_i = DiffPointRasterisation.raster_project_pullback!(ds_dout[i], points, rotation[i], translation[i], background[i], weight[i])
        push!(ds_dpoints, ds_dargs_i.points)
        @test ds_dargs_threaded.rotation[i] ≈ ds_dargs_i.rotation
        @test ds_dargs_threaded.translation[i] ≈ ds_dargs_i.translation
        @test ds_dargs_threaded.background[i] ≈ ds_dargs_i.background
        @test ds_dargs_threaded.weight[i] ≈ ds_dargs_i.weight
    end
    @test ds_dargs_threaded.points ≈ sum(ds_dpoints)
end

_pullback_alloc_serial(args, prealloc) = _pullback_alloc_points_serial(args, prealloc)

function _pullback_alloc_threaded(args, prealloc, n)
    points = _pullback_alloc_points_threaded(args, prealloc, n)
    other_args = Base.structdiff(args, NamedTuple{(:points,)})
    others = _pullback_alloc_others_threaded(other_args, prealloc)
    merge(points, others)
end

function _pullback_alloc_others_threaded(need_allocation, ::NamedTuple{})
    keys_alloc = prefix.(keys(need_allocation))
    vals = make_similar.(values(need_allocation))
    NamedTuple{keys_alloc}(vals)
end

function _pullback_alloc_others_threaded(args, prealloc)
    # it's a bit tricky to get this type-stable, but the following does the trick
    need_allocation = Base.structdiff(args, prealloc)
    keys_alloc = prefix.(keys(need_allocation))
    vals = make_similar.(values(need_allocation))
    alloc = NamedTuple{keys_alloc}(vals)
    keys_prealloc = prefix.(keys(prealloc))
    prefixed_prealloc = NamedTuple{keys_prealloc}(values(prealloc))
    merge(prefixed_prealloc, alloc)
end

_pullback_alloc_points_serial(args, prealloc) = (;ds_dpoints = get(() -> similar(args.points), prealloc, :points))

_pullback_alloc_points_threaded(args, prealloc, n) = (;ds_dpoints = get(() -> [similar(args.points) for _ in 1:n], prealloc, :points))

prefix(s::Symbol) = Symbol("ds_d" * string(s))

make_similar(x) = similar(x)
make_similar(x::AbstractVector{<:AbstractArray}) = [similar(element) for element in x]


function interpolation_weight(n, N, deltas, shift)
    val = @inbounds shift[n] == 1 ? one(eltype(deltas)) : -one(eltype(deltas))
    # loop over other dimensions
    @inbounds for other_n in 1:N
        if n == other_n
            continue
        end
        val *= deltas[other_n, mod1(shift[other_n], 2)]
    end
    val
end

"""
    digitstuple(k, Val(N))

return a N-tuple containing the bit-representation of k
"""
digitstuple(k, ::Val{N}) where {N} = ntuple(i -> k>>(i-1) % 2, N)

@testitem "digitstuple" begin
    @test DiffPointRasterisation.digitstuple(5, Val(3)) == (1, 0, 1) 
    @test DiffPointRasterisation.digitstuple(2, Val(2)) == (0, 1) 
    @test DiffPointRasterisation.digitstuple(2, Val(4)) == (0, 1, 0, 0) 
end

voxel_shifts(::Val{N}) where {N} = ntuple(k -> digitstuple(k-1, Val(N)), 2^N)