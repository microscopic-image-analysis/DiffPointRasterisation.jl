using StaticArrays
using CUDA: allowscalar,CuArray
allowscalar(false)
using KernelAbstractions
using LinearAlgebra: norm2,×

function svector_kernel(;N = (100,100,100), f=Array)
    p = zeros(Float32,N) |> f;
    u = rand(Float32,N...,3) |> f;
    @kernel function kern(p,u)
        I = @index(Global,Cartesian)
        v1 = SVector{3}(I.I)
        v2 = SVector{3}(@view u[I,:])
        p[I] = norm2(v1 × v2)
    end
    apply!(p,u) = kern(get_backend(p),64)(p,u,ndrange=size(p))
    apply!(p,u)
    p
end