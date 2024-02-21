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
    ds_dout::AbstractArray{<:Number, N_out},
    points,
    rotation::AbstractArray{<:Number, N_rotation},
    args...;
    prealloc...
) where {N_out, N_rotation} = _raster_pullback!(
    Val(N_out - (N_rotation - 2)),
    ds_dout,
    points,
    rotation,
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
    ds_dout::AbstractArray{<:Number, N_out},
    points,
    rotation::AbstractArray{<:Number, N_rotation},
    args...;
    prealloc...
) where {N_out, N_rotation} = _raster_pullback!(
    Val(N_out + 1 - (N_rotation - 2)),
    ds_dout,
    points,
    rotation,
    args...;
    prealloc...
)

@kernel function raster_pullback_kernel!(
    ::Val{N_in},
    ds_dout::AbstractArray{T, N_out_p1},
    @Const(points),
    @Const(rotations),
    @Const(translations),
    @Const(weights),
    @Const(shifts),
    @Const(scale),
    @Const(projection_idxs),
    # outputs:
    ds_dpoints,
    ds_dprojection_rotation,
    ds_dtranslation,
    ds_dweight,

) where {T, N_in, N_out_p1}
    @uniform N_out = N_out_p1 - 1  # dimensionality of output, without batch dimension
    neighbor_voxel_id, point_idx, batch_idx = @index(Global, NTuple)
    s, _, b = @index(Local, NTuple)
    @uniform n_voxel, points_per_workgroup, batchsize_per_workgroup = @groupsize()
    @assert points_per_workgroup == 1
    @assert n_voxel == 2^N_out

    ds_dpoint_rot = @localmem T (N_out, n_voxel, batchsize_per_workgroup)
    ds_dweight_private = @private T 1

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
        @inbounds ds_dweight_private[1] = voxel_weight(
            deltas,
            shift,
            projection_idxs,
            ds_dout[voxel_idx],
        )

        factor = ds_dout[voxel_idx] * weight
        ds_dcoord_part = SVector(factor .* ntuple(n -> interpolation_weight(n, N_out, deltas, shift), N_out))
        @inbounds ds_dpoint_rot[:, s, b] = ds_dcoord_part .* scale
    else
        @inbounds ds_dweight_private[1] = zero(T)
        @inbounds ds_dpoint_rot[:, s, b] .= zero(T)
    end

    @inbounds Atomix.@atomic ds_dweight[batch_idx] += ds_dweight_private[1]

    # parallel summation of ds_dpoint_rot over neighboring-voxel dimension
    @private stride = 1
    @inbounds while stride < n_voxel
        @synchronize()
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

    @synchronize()
    i = @index(Local, Linear)
    while i <= N_out
        val = ds_dpoint_rot[i, 1, b]
        @inbounds Atomix.@atomic ds_dtranslation[i, batch_idx] += val
        coef = ds_dpoint_rot[i, 1, b]
        j = 1
        while j <= N_in
            val = coef * point[j]
            @inbounds Atomix.@atomic ds_dprojection_rotation[i, j, batch_idx] += val 
            j += 1
        end
        i += prod(@groupsize())
    end

    # parallel summation of ds_dpoint_rot over batch dimension
    stride = 1
    @inbounds while stride < batchsize_per_workgroup
        @synchronize()
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

    @synchronize()
    i = @index(Local, Linear)
    while i <= N_in
        val = zero(T)
        j = 1
        while j <= N_out
            val += rotation[j, i] * ds_dpoint_rot[j, 1, 1]
            j += 1
        end
        @inbounds ds_dpoints[i, point_idx] = val
        i += prod(@groupsize())
    end

    nothing
end


function _raster_pullback_ka!(
    ::Val{N_in},
    ds_dout::AbstractArray{T, N_out_p1},
    points::AbstractMatrix{<:Number},
    rotation::AbstractArray{<:Number, 3},
    translation::AbstractMatrix{<:Number},
    # TODO: for some reason type inference fails if the following
    # two arrays are FillArrays... 
    background::AbstractVector{<:Number}=zeros(T, size(rotation, 3)),
    weight::AbstractVector{<:Number}=ones(T, size(rotation, 3)),
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

    backend = get_backend(ds_dout)
    ndrange = (2^N_out, n_points, batch_size)
    workgroup_size = (2^N_out, 1, 1024 ÷ (2^N_out))

    raster_pullback_kernel!(backend, workgroup_size, ndrange)(args...)

    ds_drotation = N_out == N_in ? ds_dprojection_rotation : vcat(ds_dprojection_rotation, KernelAbstractions.zeros(backend, T, 1, N_in, batch_size))
    ds_dbackground = dropdims(sum(ds_dout; dims=1:N_out); dims=1:N_out)
    synchronize(backend)

    return (;
        points=ds_dpoints,
        rotation=ds_drotation,
        translation=ds_dtranslation,
        background=ds_dbackground,
        weight=ds_dweight,
    )
end

@testitem "_raster_pullback_ka!" begin
    using BenchmarkTools, Rotations
    include("testing.jl")

    batch_size = batch_size_for_test()

    ds_dout = randn(8, 8, 8, batch_size)
    points = 0.3 .* randn(3, 1000)
    rotation = stack(rand(QuatRotation, batch_size))
    translation = zeros(3, batch_size)
    background = zeros(batch_size)
    weight = ones(batch_size)

    args = (Val(3), ds_dout, points, rotation, translation, background, weight)
    ds_dargs_threaded = DiffPointRasterisation._raster_pullback!(args...)
    ds_dargs_ka = DiffPointRasterisation._raster_pullback_ka!(args...)

    for prop in propertynames(ds_dargs_threaded)
        @test getproperty(ds_dargs_ka, prop) ≈ getproperty(ds_dargs_threaded, prop)
    end
end


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
) where {N_in, N_out, T<:Number}
    # The strategy followed here is to redo some of the calculations
    # made in the forward pass instead of caching them in the forward
    # pass and reusing them here.
    args = (;points,)
    @unpack ds_dpoints = _pullback_alloc_serial(args, NamedTuple(prealloc))
    accumulate_prealloc || fill!(ds_dpoints, zero(T))

    origin = (-@SVector ones(T, N_out)) - translation
    projection_idxs = SVector(ntuple(identity, N_out))
    scale = SVector{N_out, T}(size(ds_dout)) / 2 
    shifts=voxel_shifts(Val(N_out))
    all_density_idxs = CartesianIndices(ds_dout)

    # initialize some output for accumulation
    ds_dtranslation = @SVector zeros(T, N_out)
    ds_dprojection_rotation = @SMatrix zeros(T, N_out, N_in)
    ds_dweight = zero(T)

    # loop over points
    for (pt_idx, point) in enumerate(eachcol(points))
        point = SVector{N_in, T}(point)
        coord_reference_voxel, deltas = reference_coordinate_and_deltas(
            point,
            rotation,
            projection_idxs,
            origin,
            scale,
        )

        ds_dcoord = @SVector zeros(T, N_out)
        # loop over voxels that are affected by point
        for shift in shifts
            voxel_idx = CartesianIndex(Tuple(coord_reference_voxel)) + CartesianIndex(shift)
            (voxel_idx in all_density_idxs) || continue

            ds_dweight += voxel_weight(
                deltas,
                shift,
                projection_idxs,
                ds_dout[voxel_idx],
            )

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
    ds_dout::AbstractArray{T},
    points::AbstractMatrix{<:Number},
    rotation::AbstractArray{<:Number, 3},
    translation::AbstractMatrix{<:Number},
    # TODO: for some reason type inference fails if the following
    # two arrays are FillArrays... 
    background::AbstractVector{<:Number}=zeros(T, size(rotation, 3)),
    weight::AbstractVector{<:Number}=ones(T, size(rotation, 3));
    prealloc...
) where {N_in, T<:Number}
    out_batch_dim = ndims(ds_dout)
    batch_axis = axes(ds_dout, out_batch_dim)
    @argcheck axes(ds_dout, out_batch_dim) == axes(rotation, 3) == axes(translation, 2) == axes(background, 1) == axes(weight, 1)
    args = (;points, rotation, translation, background, weight)
    @unpack ds_dpoints, ds_drotation, ds_dtranslation, ds_dbackground, ds_dweight = _pullback_alloc_threaded(args, NamedTuple(prealloc), min(length(batch_axis), Threads.nthreads()))
    @assert ndims(ds_dpoints) == 3
    fill!(ds_dpoints, zero(T))

    Threads.@threads for (idxs, ichunk) in chunks(batch_axis, size(ds_dpoints, 3))
        for i in idxs
            args_i = (selectdim(ds_dout, out_batch_dim, i), points, view(rotation, :, :, i), view(translation, :, i), background[i], weight[i])
            result_i = _raster_pullback!(Val(N_in), args_i...; accumulate_prealloc=true, points=view(ds_dpoints, :, :, ichunk))
            ds_drotation[:, :, i] .= result_i.rotation
            ds_dtranslation[:, i] = result_i.translation
            ds_dbackground[i] = result_i.background
            ds_dweight[i] = result_i.weight
        end
    end
    return (; points=dropdims(sum(ds_dpoints; dims=3); dims=3), rotation=ds_drotation, translation=ds_dtranslation, background=ds_dbackground, weight=ds_dweight)
end

@testitem "raster_pullback! threaded" begin
    include("../test/data.jl")

    ds_dout = randn(D.grid_size_3d..., D.batch_size)

    ds_dargs_threaded = DiffPointRasterisation.raster_pullback!(ds_dout, D.more_points, D.rotations, D.translations_3d, D.backgrounds, D.weights)

    ds_dpoints = Matrix{Float64}[]
    for i in 1:D.batch_size
        ds_dargs_i = @views raster_pullback!(ds_dout[:, :, :, i], D.more_points, D.rotations[:, :, i], D.translations_3d[:, i], D.backgrounds[i], D.weights[i])
        push!(ds_dpoints, ds_dargs_i.points)
        @views begin
            @test ds_dargs_threaded.rotation[:, :, i] ≈ ds_dargs_i.rotation
            @test ds_dargs_threaded.translation[:, i] ≈ ds_dargs_i.translation
            @test ds_dargs_threaded.background[i] ≈ ds_dargs_i.background
            @test ds_dargs_threaded.weight[i] ≈ ds_dargs_i.weight
        end
    end
    @test ds_dargs_threaded.points ≈ sum(ds_dpoints)
end

@testitem "raster_project_pullback! threaded" begin
    include("../test/data.jl")

    ds_dout = zeros(D.grid_size_2d..., D.batch_size)

    ds_dargs_threaded = DiffPointRasterisation.raster_project_pullback!(ds_dout, D.more_points, D.rotations, D.translations_2d, D.backgrounds, D.weights)

    ds_dpoints = Matrix{Float64}[]
    for i in 1:D.batch_size
        ds_dargs_i = @views raster_project_pullback!(ds_dout[:, :, i], D.more_points, D.rotations[:, :, i], D.translations_2d[:, i], D.backgrounds[i], D.weights[i])
        push!(ds_dpoints, ds_dargs_i.points)
        @views begin
            @test ds_dargs_threaded.rotation[:, :, i] ≈ ds_dargs_i.rotation
            @test ds_dargs_threaded.translation[:, i] ≈ ds_dargs_i.translation
            @test ds_dargs_threaded.background[i] ≈ ds_dargs_i.background
            @test ds_dargs_threaded.weight[i] ≈ ds_dargs_i.weight
        end
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
    vals = similar.(values(need_allocation))
    NamedTuple{keys_alloc}(vals)
end

function _pullback_alloc_others_threaded(args, prealloc)
    # it's a bit tricky to get this type-stable, but the following does the trick
    need_allocation = Base.structdiff(args, prealloc)
    keys_alloc = prefix.(keys(need_allocation))
    vals = similar.(values(need_allocation))
    alloc = NamedTuple{keys_alloc}(vals)
    keys_prealloc = prefix.(keys(prealloc))
    prefixed_prealloc = NamedTuple{keys_prealloc}(values(prealloc))
    merge(prefixed_prealloc, alloc)
end

_pullback_alloc_points_serial(args, prealloc) = (;ds_dpoints = get(() -> similar(args.points), prealloc, :points))

_pullback_alloc_points_threaded(args, prealloc, n) = (;ds_dpoints = get(() -> similar(args.points, (size(args.points)..., n)), prealloc, :points))


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