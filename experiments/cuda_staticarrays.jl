using CUDA
using Rotations
using StaticArrays


function rotate(R::SMatrix, points::CuMatrix)
    function rotate_kernel(out, q, points)

        nothing
    end
    out = similar(points)
end