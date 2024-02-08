function batch_size_for_test()
    local batch_size = Threads.nthreads() + 1
    while (Threads.nthreads() > 1) && (batch_size % Threads.nthreads() == 0)
        batch_size += 1
    end
    batch_size
end


function run_cuda(f::F, args::Vararg{Any, N}) where {F, N}
    cu_args = cu.(args)
    return f(cu_args...)
end