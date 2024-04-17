# single image
function raster_pullback!(
    ds_dout::AbstractArray{<:Number, N_out},
    points::AbstractVector{<:StaticVector{N_in, <:Number}},
    rotation::StaticMatrix{N_out, N_in, TR},
    translation::StaticVector{N_out, TT},
    background::Number,
    out_weight::OW,
    point_weight::AbstractVector{<:Number},
    ds_dpoints::AbstractMatrix{TP},
    ds_dpoint_weight::AbstractVector{PW};
    accumulate_ds_dpoints=false,
) where {N_in, N_out, TP<:Number, TR<:Number, TT<:Number, OW<:Number, PW<:Number}
    T = promote_type(eltype(ds_dout), TP, TR, TT, OW, PW)
    @argcheck size(ds_dpoints, 1) == N_in
    @argcheck length(point_weight) == length(points) == length(ds_dpoint_weight) == size(ds_dpoints, 2)
    # The strategy followed here is to redo some of the calculations
    # made in the forward pass instead of caching them in the forward
    # pass and reusing them here.
    if !accumulate_ds_dpoints
        fill!(ds_dpoints, zero(TP))
        fill!(ds_dpoint_weight, zero(PW))
    end

    origin = (-@SVector ones(TT, N_out)) - translation
    scale = SVector{N_out, T}(size(ds_dout)) / 2 
    shifts=voxel_shifts(Val(N_out))
    all_density_idxs = CartesianIndices(ds_dout)

    # initialize some output for accumulation
    ds_dtranslation = @SVector zeros(TT, N_out)
    ds_drotation = @SMatrix zeros(TR, N_out, N_in)
    ds_dout_weight = zero(OW)

    # loop over points
    for (pt_idx, point) in enumerate(points)
        point = SVector{N_in, TP}(point)
        point_weight_i = point_weight[pt_idx]
        coord_reference_voxel, deltas = reference_coordinate_and_deltas(
            point,
            rotation,
            origin,
            scale,
        )

        ds_dcoord = @SVector zeros(T, N_out)
        ds_dpoint_weight_i = zero(PW)
        # loop over voxels that are affected by point
        for shift in shifts
            voxel_idx = CartesianIndex(Tuple(coord_reference_voxel)) + CartesianIndex(shift)
            (voxel_idx in all_density_idxs) || continue

            ds_dout_i = ds_dout[voxel_idx]

            ds_dweight = voxel_weight(
                deltas,
                shift,
                ds_dout_i,
            )

            ds_dout_weight += ds_dweight * point_weight_i
            ds_dpoint_weight_i += ds_dweight * out_weight

            factor = ds_dout_i * out_weight * point_weight_i
            # loop over dimensions of point
            ds_dcoord += SVector(factor .* ntuple(n -> interpolation_weight(n, N_out, deltas, shift), Val(N_out)))
        end
        scaled = ds_dcoord .* scale
        ds_dtranslation += scaled
        ds_drotation += scaled * point'
        ds_dpoint = rotation' * scaled 
        @view(ds_dpoints[:, pt_idx]) .+= ds_dpoint
        ds_dpoint_weight[pt_idx] += ds_dpoint_weight_i
    end
    return (; points=ds_dpoints, rotation=ds_drotation, translation=ds_dtranslation, background=sum(ds_dout), out_weight=ds_dout_weight, point_weight=ds_dpoint_weight)
end

# batch of images
function raster_pullback!(
    ds_dout::AbstractArray{<:Number, N_out_p1},
    points::AbstractVector{<:StaticVector{N_in, <:Number}},
    rotation::AbstractVector{<:StaticMatrix{N_out, N_in, <:Number}},
    translation::AbstractVector{<:StaticVector{N_out, <:Number}},
    background::AbstractVector{<:Number},
    out_weight::AbstractVector{<:Number},
    point_weight::AbstractVector{<:Number},
    ds_dpoints::AbstractArray{<:Number, 3},
    ds_drotation::AbstractArray{<:Number, 3},
    ds_dtranslation::AbstractMatrix{<:Number},
    ds_dbackground::AbstractVector{<:Number},
    ds_dout_weight::AbstractVector{<:Number},
    ds_dpoint_weight::AbstractMatrix{<:Number},
) where {N_in, N_out, N_out_p1}
    batch_axis = axes(ds_dout, N_out_p1)
    @argcheck N_out == N_out_p1 - 1
    @argcheck batch_axis == axes(rotation, 1) == axes(translation, 1) == axes(background, 1) == axes(out_weight, 1)
    @argcheck batch_axis == axes(ds_drotation, 3) == axes(ds_dtranslation, 2) == axes(ds_dbackground, 1) == axes(ds_dout_weight, 1)
    fill!(ds_dpoints, zero(eltype(ds_dpoints)))
    fill!(ds_dpoint_weight, zero(eltype(ds_dpoint_weight)))

    n_threads = size(ds_dpoints, 3)

    Threads.@threads for (idxs, ichunk) in chunks(batch_axis, n_threads)
        for i in idxs
            args_i = (selectdim(ds_dout, N_out_p1, i), points, rotation[i], translation[i], background[i], out_weight[i], point_weight)
            result_i = raster_pullback!(args_i..., view(ds_dpoints, :, :, ichunk), view(ds_dpoint_weight, :, ichunk); accumulate_ds_dpoints=true)
            ds_drotation[:, :, i] .= result_i.rotation
            ds_dtranslation[:, i] = result_i.translation
            ds_dbackground[i] = result_i.background
            ds_dout_weight[i] = result_i.out_weight
        end
    end
    return (; points=dropdims(sum(ds_dpoints; dims=3); dims=3), rotation=ds_drotation, translation=ds_dtranslation, background=ds_dbackground, out_weight=ds_dout_weight, point_weight=dropdims(sum(ds_dpoint_weight; dims=2); dims=2))
end


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
    ds_dpoint_weights = similar(D.point_weights)
    ds_dpoint_weights_batched = similar(D.point_weights, (size(D.point_weights)..., Threads.nthreads()))

    args_batched_3d = (
        ds_dout_3d_batched,
        D.points_static,
        D.rotations_static,
        D.translations_3d_static,
        D.backgrounds,
        D.weights,
        D.point_weights,
        ds_dpoints_batched,
        ds_drotations,
        ds_dtranslations_3d,
        ds_dbackgrounds,
        ds_dweights,
        ds_dpoint_weights_batched,
    )
    args_batched_2d = (
        ds_dout_2d_batched,
        D.points_static,
        D.projections_static,
        D.translations_2d_static,
        D.backgrounds,
        D.weights,
        D.point_weights,
        ds_dpoints_batched,
        ds_dprojections,
        ds_dtranslations_2d,
        ds_dbackgrounds,
        ds_dweights,
        ds_dpoint_weights_batched,
    )

    function to_cuda(args)
        args_cu = adapt(CuArray, args)
        args_cu = Base.setindex(args_cu, args_cu[8][:, :, 1], 8)  # ds_dpoint without batch dim
        args_cu = Base.setindex(args_cu, args_cu[13][:, 1], 13)  # ds_dpoint_weight without batch dim
    end

    # check type stability
    # single image
    @inferred DiffPointRasterisation.raster_pullback!(ds_dout_3d, D.points_static, D.rotation, D.translation_3d, D.background, D.weight, D.point_weights, ds_dpoints, ds_dpoint_weights)
    @inferred DiffPointRasterisation.raster_pullback!(ds_dout_2d, D.points_static, D.projection, D.translation_2d, D.background, D.weight, D.point_weights, ds_dpoints, ds_dpoint_weights)
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
        $(D.point_weights),
        $ds_dpoints,
        $ds_dpoint_weights,
    ) evals=1 samples=1
    @test allocations == 0
end


@testitem "raster_pullback! threaded" begin
    include("../test/data.jl")

    ds_dout = randn(D.grid_size_3d..., D.batch_size)

    ds_dargs_threaded = DiffPointRasterisation.raster_pullback!(ds_dout, D.more_points, D.rotations, D.translations_3d, D.backgrounds, D.weights, D.more_point_weights)

    ds_dpoints = Matrix{Float64}[]
    ds_dpoint_weight = Vector{Float64}[]
    for i in 1:D.batch_size
        ds_dargs_i = @views raster_pullback!(ds_dout[:, :, :, i], D.more_points, D.rotations[i], D.translations_3d[i], D.backgrounds[i], D.weights[i], D.more_point_weights)
        push!(ds_dpoints, ds_dargs_i.points)
        push!(ds_dpoint_weight, ds_dargs_i.point_weight)
        @views begin
            @test ds_dargs_threaded.rotation[:, :, i] ≈ ds_dargs_i.rotation
            @test ds_dargs_threaded.translation[:, i] ≈ ds_dargs_i.translation
            @test ds_dargs_threaded.background[i] ≈ ds_dargs_i.background
            @test ds_dargs_threaded.out_weight[i] ≈ ds_dargs_i.out_weight
        end
    end
    @test ds_dargs_threaded.points ≈ sum(ds_dpoints)
    @test ds_dargs_threaded.point_weight ≈ sum(ds_dpoint_weight)

    ds_dout = zeros(D.grid_size_2d..., D.batch_size)

    ds_dargs_threaded = DiffPointRasterisation.raster_pullback!(ds_dout, D.more_points, D.projections, D.translations_2d, D.backgrounds, D.weights, D.more_point_weights)

    ds_dpoints = Matrix{Float64}[]
    ds_dpoint_weight = Vector{Float64}[]
    for i in 1:D.batch_size
        ds_dargs_i = @views raster_pullback!(ds_dout[:, :, i], D.more_points, D.projections[i], D.translations_2d[i], D.backgrounds[i], D.weights[i], D.more_point_weights)
        push!(ds_dpoints, ds_dargs_i.points)
        push!(ds_dpoint_weight, ds_dargs_i.point_weight)
        @views begin
            @test ds_dargs_threaded.rotation[:, :, i] ≈ ds_dargs_i.rotation
            @test ds_dargs_threaded.translation[:, i] ≈ ds_dargs_i.translation
            @test ds_dargs_threaded.background[i] ≈ ds_dargs_i.background
            @test ds_dargs_threaded.out_weight[i] ≈ ds_dargs_i.out_weight
        end
    end
    @test ds_dargs_threaded.points ≈ sum(ds_dpoints)
    @test ds_dargs_threaded.point_weight ≈ sum(ds_dpoint_weight)
end