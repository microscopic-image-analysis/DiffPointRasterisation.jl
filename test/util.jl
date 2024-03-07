function run_cuda(f, args...)
    cu_args = adapt(CuArray, args)
    return f(cu_args...)
end


function cuda_cpu_agree(f, args...)
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
        try
            actual_elem = getproperty(actual_cpu, prop)
            expected_elem = getproperty(expected, prop)
            if !(actual_elem ≈ expected_elem)
                throw("Values differ:\nActual: $(string(actual_elem)) \nExpected: $(string(expected_elem))")
                return false
            end
        catch e
            println("Error while trying to compare element $(string(prop))")
            rethrow()
        end
    end
    true
end