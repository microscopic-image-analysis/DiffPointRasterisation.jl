    function batch_size_for_test()
        local batch_size = Threads.nthreads() + 1
        while (Threads.nthreads() > 1) && (batch_size % Threads.nthreads() == 0)
            batch_size += 1
        end
        batch_size
    end