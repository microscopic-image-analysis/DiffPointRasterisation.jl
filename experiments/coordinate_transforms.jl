using CoordinateTransformations, Rotations, LinearAlgebra

R = LinearMap(QuatRotation(normalize!(randn(4))))
P = LinearMap(
        [
        1. 0. 0.
        0. 1. 0.
    ]
)
t = Translation(randn(2))

affine_trans = t ∘ P ∘ R

affine_trans(randn(3))