using TestItemRunner: @run_package_tests

@run_package_tests # filter=ti-> occursin("CUDA", ti.name)