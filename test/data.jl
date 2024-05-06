module D

using Rotations, StaticArrays

function batch_size_for_test()
    local batch_size = Threads.nthreads() + 1
    while (Threads.nthreads() > 1) && (batch_size % Threads.nthreads() == 0)
        batch_size += 1
    end
    return batch_size
end

const P = @SMatrix Float64[
    1 0 0
    0 1 0
]

const grid_size_3d = (8, 8, 8)
const grid_size_2d = (8, 8)
const batch_size = batch_size_for_test()

const points = [0.4 * randn(3) for _ in 1:10]
const points_static = SVector{3}.(points)
const points_array = Matrix{Float64}(undef, 3, length(points))
eachcol(points_array) .= points
const points_reinterp = reinterpret(reshape, SVector{3,Float64}, points_array)
const more_points = [0.4 * @SVector randn(3) for _ in 1:100_000]

const rotation = rand(RotMatrix3{Float64})
const rotations_static = rand(RotMatrix3{Float64}, batch_size)::Vector{<:StaticMatrix}
const rotations = (Array.(rotations_static))::Vector{Matrix{Float64}}
const rotations_array = Array{Float64,3}(undef, 3, 3, batch_size)
eachslice(rotations_array; dims=3) .= rotations
const rotations_reinterp = reinterpret(
    reshape, SMatrix{3,3,Float64,9}, reshape(rotations_array, 9, :)
)
const rotation_tangent = Array(rand(RotMatrix3))
const rotation_tangents_static =
    rand(RotMatrix3{Float64}, batch_size)::Vector{<:StaticMatrix}
const rotation_tangents = (Array.(rotation_tangents_static))::Vector{Matrix{Float64}}

const projection = P * rand(RotMatrix3)
const projections_static = Ref(P) .* rand(RotMatrix3{Float64}, batch_size)
const projections = (Array.(projections_static))::Vector{Matrix{Float64}}
const projections_array = Array{Float64,3}(undef, 2, 3, batch_size)
eachslice(projections_array; dims=3) .= projections
const projections_reinterp = reinterpret(
    reshape, SMatrix{2,3,Float64,6}, reshape(projections_array, 6, :)
)
const projection_tangent = Array(P * rand(RotMatrix3))
const projection_tangents_static = Ref(P) .* rand(RotMatrix3{Float64}, batch_size)
const projection_tangents = (Array.(projection_tangents_static))::Vector{Matrix{Float64}}

const translation_3d = 0.1 * @SVector randn(3)
const translation_2d = 0.1 * @SVector randn(2)
const translations_3d_static = [0.1 * @SVector randn(3) for _ in 1:batch_size]
const translations_3d = (Array.(translations_3d_static))::Vector{Vector{Float64}}
const translations_3d_array = Matrix{Float64}(undef, 3, batch_size)
eachcol(translations_3d_array) .= translations_3d
const translations_3d_reinterp = reinterpret(
    reshape, SVector{3,Float64}, translations_3d_array
)
const translations_2d_static = [0.1 * @SVector randn(2) for _ in 1:batch_size]
const translations_2d = (Array.(translations_2d_static))::Vector{Vector{Float64}}
const translations_2d_array = Matrix{Float64}(undef, 2, batch_size)
eachcol(translations_2d_array) .= translations_2d
const translations_2d_reinterp = reinterpret(
    reshape, SVector{2,Float64}, translations_2d_array
)

const background = 0.1
const backgrounds = collect(1:1.0:batch_size)

const weight = rand()
const weights = 10 .* rand(batch_size)

const point_weights = let
    w = rand(length(points))
    w ./ sum(w)
end
const more_point_weights = let
    w = rand(length(more_points))
    w ./ sum(w)
end

end  # module D