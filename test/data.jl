module D

using Rotations, StaticArrays

function batch_size_for_test()
    local batch_size = Threads.nthreads() + 1
    while (Threads.nthreads() > 1) && (batch_size % Threads.nthreads() == 0)
        batch_size += 1
    end
    batch_size
end

const P = @SMatrix Float64[
    1 0 0
    0 1 0
]

const grid_size_3d = (8, 8, 8)
const grid_size_2d = (8, 8)
const batch_size = batch_size_for_test()

const points_vec = [0.4 * @SVector randn(3) for _ in 1:10]
const points = stack(points_vec)::Matrix{Float64}
const more_points_vec = [0.4 * @SVector randn(3) for _ in 1:100_000]
const more_points = stack(points_vec)::Matrix{Float64}

const rotation = Array(rand(RotMatrix3))
const rotations_vec = rand(QuatRotation{Float64}, batch_size)::Vector{<:StaticMatrix}
const rotations = stack(rotations_vec)
const rotation_tangent = Array(rand(RotMatrix3))
const rotation_tangents = stack(rand(RotMatrix3, batch_size))

const projection = Array(P * rand(RotMatrix3))
const projections_vec = Ref(P) .* rand(QuatRotation{Float64}, batch_size)
const projections = stack(projections_vec)

const translation_3d = 0.1 * randn(3)
const translation_2d = 0.1 * randn(2)
const translations_3d_vec = [0.1 * @SVector randn(3) for _ in 1:batch_size]
const translations_3d = stack(translations_3d_vec)::Matrix{Float64}
const translations_2d_vec = [0.1 * @SVector randn(2) for _ in 1:batch_size]
const translations_2d = stack(translations_2d_vec)::Matrix{Float64}

const background = 0.1
const backgrounds = collect(1:1.0:batch_size)

const weight = rand()
const weights = 10 .* rand(batch_size)

end  # module D