module DiffPointRasterisationChainRulesCoreExt

using DiffPointRasterisation, ChainRulesCore, StaticArrays

# single image
function ChainRulesCore.rrule(
    ::typeof(DiffPointRasterisation.raster),
    grid_size,
    points::AbstractVector{<:StaticVector{N_in,T}},
    rotation::AbstractMatrix{<:Number},
    translation::AbstractVector{<:Number},
    optional_args...,
) where {N_in,T<:Number}
    out = raster(grid_size, points, rotation, translation, optional_args...)

    function raster_pullback(ds_dout)
        out_pb = raster_pullback!(
            unthunk(ds_dout), points, rotation, translation, optional_args...
        )
        ds_dpoints = reinterpret(reshape, SVector{N_in,T}, out_pb.points)
        return NoTangent(),
        NoTangent(), ds_dpoints,
        values(out_pb)[2:(3 + length(optional_args))]...
    end

    return out, raster_pullback
end

function ChainRulesCore.rrule(
    f::typeof(DiffPointRasterisation.raster),
    grid_size,
    points::AbstractVector{<:AbstractVector{<:Number}},
    rotation::AbstractMatrix{<:Number},
    translation::AbstractVector{<:Number},
    optional_args...,
)
    return ChainRulesCore.rrule(
        f,
        grid_size,
        DiffPointRasterisation.inner_to_sized(points),
        rotation,
        translation,
        optional_args...,
    )
end

# batch of images
function ChainRulesCore.rrule(
    ::typeof(DiffPointRasterisation.raster),
    grid_size,
    points::AbstractVector{<:StaticVector{N_in,TP}},
    rotation::AbstractVector{<:StaticMatrix{N_out,N_in,TR}},
    translation::AbstractVector{<:StaticVector{N_out,TT}},
    optional_args...,
) where {N_in,N_out,TP<:Number,TR<:Number,TT<:Number}
    out = raster(grid_size, points, rotation, translation, optional_args...)

    function raster_pullback(ds_dout)
        out_pb = raster_pullback!(
            unthunk(ds_dout), points, rotation, translation, optional_args...
        )
        ds_dpoints = reinterpret(reshape, SVector{N_in,TP}, out_pb.points)
        L = N_out * N_in
        ds_drotation = reinterpret(
            reshape, SMatrix{N_out,N_in,TR,L}, reshape(out_pb.rotation, L, :)
        )
        ds_dtranslation = reinterpret(reshape, SVector{N_out,TT}, out_pb.translation)
        return NoTangent(),
        NoTangent(), ds_dpoints, ds_drotation, ds_dtranslation,
        values(out_pb)[4:(3 + length(optional_args))]...
    end

    return out, raster_pullback
end

function ChainRulesCore.rrule(
    f::typeof(DiffPointRasterisation.raster),
    grid_size,
    points::AbstractVector{<:AbstractVector{<:Number}},
    rotation::AbstractVector{<:AbstractMatrix{<:Number}},
    translation::AbstractVector{<:AbstractVector{<:Number}},
    optional_args...,
)
    return ChainRulesCore.rrule(
        f,
        grid_size,
        DiffPointRasterisation.inner_to_sized(points),
        DiffPointRasterisation.inner_to_sized(rotation),
        DiffPointRasterisation.inner_to_sized(translation),
        optional_args...,
    )
end

end  # module DiffPointRasterisationChainRulesCoreExt