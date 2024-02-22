function run_cuda(f::F, args::Vararg{Any, N}) where {F, N}
    cu_args = adapt(CuArray, args)
    return f(cu_args...)
end


function cuda_cpu_agree(f::F, args...) where {F}
    out_cpu = f(args...)
    out_cuda = run_cuda(f, args...)
    is_approx_equal(out_cuda, out_cpu)
end

function is_approx_equal(actual::AbstractArray, expected::AbstractArray)
    Array(actual) ≈ expected
end


function is_approx_equal(actual::NamedTuple, expected::NamedTuple)
    actual_cpu = adapt(Array, actual)
    for prop in propertynames(expected)
        # (prop in (:points,)) && continue
        actual_elem = getproperty(actual_cpu, prop)
        expected_elem = getproperty(expected, prop)
        if !(actual_elem ≈ expected_elem)
            throw("Element '$(string(prop))' differs:\nActual: $actual_elem \nExpected: $expected_elem")
            return false
        end
    end
    true
end