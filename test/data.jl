module D

using Rotations

function batch_size_for_test()
    local batch_size = Threads.nthreads() + 1
    while (Threads.nthreads() > 1) && (batch_size % Threads.nthreads() == 0)
        batch_size += 1
    end
    batch_size
end


const grid_size_3d = (8, 8, 8)
const grid_size_2d = (8, 8)
const batch_size = batch_size_for_test()
const points = 0.4 .* randn(3, 10)
const more_points = 0.4 .* randn(3, 100_000)
const rotation = Array(rand(RotMatrix3))
const rotation_tangent = Array(rand(RotMatrix3))
const rotations = stack(rand(RotMatrix3, batch_size))
const rotation_tangents = stack(rand(RotMatrix3, batch_size))
const translation_3d = 0.1 .* randn(3)
const translation_2d = 0.1 .* randn(2)
const translations_3d = zeros(3, batch_size)
const translations_2d = zeros(2, batch_size)
const background = 0.1
const backgrounds = collect(1:1.0:batch_size)
const weight = rand()
const weights = 10 .* rand(batch_size)

end  # module D