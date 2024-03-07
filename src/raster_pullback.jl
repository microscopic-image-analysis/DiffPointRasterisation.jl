# single image
function raster_pullback!(
    ds_dout::AbstractArray{<:Number, N_out},
    points::AbstractVector{<:StaticVector{N_in, <:Number}},
    rotation::StaticMatrix{N_out, N_in, TR},
    translation::StaticVector{N_out, TT},
    background::Number,
    weight::TW,
    ds_dpoints::AbstractMatrix{TP};
    accumulate_ds_dpoints=false,
) where {N_in, N_out, TP<:Number, TR<:Number, TT<:Number, TW<:Number}
    T = promote_type(eltype(ds_dout), TP, TR, TT, TW)
    @argcheck size(ds_dpoints, 1) == N_in
    # The strategy followed here is to redo some of the calculations
    # made in the forward pass instead of caching them in the forward
    # pass and reusing them here.
    accumulate_ds_dpoints || fill!(ds_dpoints, zero(TP))

    origin = (-@SVector ones(TT, N_out)) - translation
    scale = SVector{N_out, T}(size(ds_dout)) / 2 
    shifts=voxel_shifts(Val(N_out))
    all_density_idxs = CartesianIndices(ds_dout)

    # initialize some output for accumulation
    ds_dtranslation = @SVector zeros(TT, N_out)
    ds_drotation = @SMatrix zeros(TR, N_out, N_in)
    ds_dweight = zero(TW)

    # loop over points
    for (pt_idx, point) in enumerate(points)
        point = SVector{N_in, TP}(point)
        coord_reference_voxel, deltas = reference_coordinate_and_deltas(
            point,
            rotation,
            origin,
            scale,
        )

        ds_dcoord = @SVector zeros(T, N_out)
        # loop over voxels that are affected by point
        for shift in shifts
            voxel_idx = CartesianIndex(Tuple(coord_reference_voxel)) + CartesianIndex(shift)
            (voxel_idx in all_density_idxs) || continue

            ds_dout_i = ds_dout[voxel_idx]

            ds_dweight += voxel_weight(
                deltas,
                shift,
                ds_dout_i,
            )

            factor = ds_dout_i * weight
            # loop over dimensions of point
            ds_dcoord += SVector(factor .* ntuple(n -> interpolation_weight(n, N_out, deltas, shift), N_out))
        end
        scaled = ds_dcoord .* scale
        ds_dtranslation += scaled
        ds_drotation += scaled * point'
        ds_dpoint = rotation' * scaled 
        @view(ds_dpoints[:, pt_idx]) .+= ds_dpoint

    end
    return (; points=ds_dpoints, rotation=ds_drotation, translation=ds_dtranslation, background=sum(ds_dout), weight=ds_dweight)
end

# batch of images
function raster_pullback!(
    ds_dout::AbstractArray{<:Number, N_out_p1},
    points::AbstractVector{<:StaticVector{N_in, <:Number}},
    rotation::AbstractVector{<:StaticMatrix{N_out, N_in, <:Number}},
    translation::AbstractVector{<:StaticVector{N_out, <:Number}},
    background::AbstractVector{<:Number},
    weight::AbstractVector{<:Number},
    ds_dpoints::AbstractArray{<:Number, 3},
    ds_drotation::AbstractArray{<:Number, 3},
    ds_dtranslation::AbstractMatrix{<:Number},
    ds_dbackground::AbstractVector{<:Number},
    ds_dweight::AbstractVector{<:Number},
) where {N_in, N_out, N_out_p1}
    batch_axis = axes(ds_dout, N_out_p1)
    @argcheck N_out == N_out_p1 - 1
    @argcheck batch_axis == axes(rotation, 1) == axes(translation, 1) == axes(background, 1) == axes(weight, 1)
    @argcheck batch_axis == axes(ds_drotation, 3) == axes(ds_dtranslation, 2) == axes(ds_dbackground, 1) == axes(ds_dweight, 1)
    fill!(ds_dpoints, zero(eltype(ds_dpoints)))

    n_threads = size(ds_dpoints, 3)

    Threads.@threads for (idxs, ichunk) in chunks(batch_axis, n_threads)
        for i in idxs
            args_i = (selectdim(ds_dout, N_out_p1, i), points, rotation[i], translation[i], background[i], weight[i])
            result_i = raster_pullback!(args_i..., view(ds_dpoints, :, :, ichunk); accumulate_ds_dpoints=true)
            ds_drotation[:, :, i] .= result_i.rotation
            ds_dtranslation[:, i] = result_i.translation
            ds_dbackground[i] = result_i.background
            ds_dweight[i] = result_i.weight
        end
    end
    return (; points=dropdims(sum(ds_dpoints; dims=3); dims=3), rotation=ds_drotation, translation=ds_dtranslation, background=ds_dbackground, weight=ds_dweight)
end


prefix(s::Symbol) = Symbol("ds_d" * string(s))


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

@testitem "raster_pullback! inference and allocations" begin
    using BenchmarkTools, CUDA, Adapt
    include("../test/data.jl")
    ds_dout_3d = randn(D.grid_size_3d)
    ds_dout_3d_batched = randn(D.grid_size_3d..., D.batch_size)
    ds_dout_2d = randn(D.grid_size_2d)
    ds_dout_2d_batched = randn(D.grid_size_2d..., D.batch_size)

    ds_dpoints = similar(D.points_array)
    ds_dpoints_batched = similar(D.points_array, (size(D.points_array)..., Threads.nthreads()))
    ds_drotations = similar(D.rotations_array)
    ds_dprojections = similar(D.projections_array)
    ds_dtranslations_3d = similar(D.translations_3d_array)
    ds_dtranslations_2d = similar(D.translations_2d_array)
    ds_dbackgrounds = similar(D.backgrounds)
    ds_dweights = similar(D.weights)

    args_batched_3d = (
        ds_dout_3d_batched,
        D.points_static,
        D.rotations_static,
        D.translations_3d_static,
        D.backgrounds,
        D.weights,
        ds_dpoints_batched,
        ds_drotations,
        ds_dtranslations_3d,
        ds_dbackgrounds,
        ds_dweights
    )
    args_batched_2d = (
        ds_dout_2d_batched,
        D.points_static,
        D.projections_static,
        D.translations_2d_static,
        D.backgrounds,
        D.weights,
        ds_dpoints_batched,
        ds_dprojections,
        ds_dtranslations_2d,
        ds_dbackgrounds,
        ds_dweights
    )

    function to_cuda(args)
        args_cu = adapt(CuArray, args)
        Base.setindex(args_cu, args_cu[7][:, :, 1], 7)  # ds_dpoint without batch dim
    end

    # check type stability
    # single image
    @inferred DiffPointRasterisation.raster_pullback!(ds_dout_3d, D.points_static, D.rotation, D.translation_3d, D.background, D.weight, ds_dpoints)
    @inferred DiffPointRasterisation.raster_pullback!(ds_dout_2d, D.points_static, D.projection, D.translation_2d, D.background, D.weight, ds_dpoints)
    # batched
    @inferred DiffPointRasterisation.raster_pullback!(args_batched_3d...)
    @inferred DiffPointRasterisation.raster_pullback!(args_batched_2d...)
    if CUDA.functional()
        cu_args_3d = to_cuda(args_batched_3d)
        @inferred DiffPointRasterisation.raster_pullback!(cu_args_3d...)
        cu_args_2d = to_cuda(args_batched_2d)
        @inferred DiffPointRasterisation.raster_pullback!(cu_args_2d...)
    end

    # check that single-imge pullback is allocation-free
    allocations = @ballocated DiffPointRasterisation.raster_pullback!(
        $ds_dout_3d,
        $(D.points_static),
        $(D.rotation),
        $(D.translation_3d),
        $(D.background),
        $(D.weight),
        $ds_dpoints,
    ) evals=1 samples=1
    @test allocations == 0
end


@testitem "raster_pullback! threaded" begin
    include("../test/data.jl")

    ds_dout = randn(D.grid_size_3d..., D.batch_size)

    ds_dargs_threaded = DiffPointRasterisation.raster_pullback!(ds_dout, D.more_points, D.rotations, D.translations_3d, D.backgrounds, D.weights)

    ds_dpoints = Matrix{Float64}[]
    for i in 1:D.batch_size
        ds_dargs_i = @views raster_pullback!(ds_dout[:, :, :, i], D.more_points, D.rotations[i], D.translations_3d[i], D.backgrounds[i], D.weights[i])
        push!(ds_dpoints, ds_dargs_i.points)
        @views begin
            @test ds_dargs_threaded.rotation[:, :, i] ≈ ds_dargs_i.rotation
            @test ds_dargs_threaded.translation[:, i] ≈ ds_dargs_i.translation
            @test ds_dargs_threaded.background[i] ≈ ds_dargs_i.background
            @test ds_dargs_threaded.weight[i] ≈ ds_dargs_i.weight
        end
    end
    @test ds_dargs_threaded.points ≈ sum(ds_dpoints)


    ds_dout = zeros(D.grid_size_2d..., D.batch_size)

    ds_dargs_threaded = DiffPointRasterisation.raster_pullback!(ds_dout, D.more_points, D.projections, D.translations_2d, D.backgrounds, D.weights)

    ds_dpoints = Matrix{Float64}[]
    for i in 1:D.batch_size
        ds_dargs_i = @views raster_pullback!(ds_dout[:, :, i], D.more_points, D.projections[i], D.translations_2d[i], D.backgrounds[i], D.weights[i])
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