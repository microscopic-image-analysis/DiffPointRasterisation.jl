module DiffPointRasterisationChainRulesCoreExt

using DiffPointRasterisation, ChainRulesCore

function ChainRulesCore.rrule(
    ::typeof(raster), 
    grid_size,
    args...,
)
    out = raster(grid_size, args...)

    raster_pullback(ds_dout) = NoTangent(), NoTangent(), values(
        raster_pullback!(
            unthunk(ds_dout),
            args...,
        )
    )[1:length(args)]...

    return out, raster_pullback
end


function ChainRulesCore.rrule(
    ::typeof(raster_project), 
    grid_size,
    args...,
)
    out = raster_project(grid_size, args...)

    raster_project_pullback(ds_dout) = NoTangent(), NoTangent(), values(
        raster_project_pullback!(
            unthunk(ds_dout),
            args...,
        )
    )[1:length(args)]...

    return out, raster_project_pullback
end

end  # module DiffPointRasterisationChainRulesCoreExt