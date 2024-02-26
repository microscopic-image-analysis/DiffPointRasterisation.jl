using DiffPointRasterisation
using FFTW
using Images
using Zygote

load_image(path) = load(path) .|> Gray |> channelview

init_points(n) = (rand(Float32, 2, n) .- 0.5f0) .* Float32[2;2;;]

target_image = load_image("data/julia.png")

points = init_points(5_000)
rotation = Float32[1;0;;0;1;;]
translation = zeros(Float32, 2)

function model(points, log_bandwidth, log_weight) 
    rough_image = raster(size(target_image), points, rotation, translation, 0f0, exp(log_weight))
    # smooth with gaussian kernel
    kernel = gaussian_kernel(log_bandwidth, size(target_image)...)
    image = convolve_image(rough_image, kernel)
    image
end

function gaussian_kernel(log_σ::T, h=4*ceil(Int, σ) + 1, w=h) where {T}
    σ = exp(log_σ)
    mw = T(0.5 * (w + 1))
    mh = T(0.5 * (h + 1))
    gw = [exp(-(x - mw)^2/(2*σ^2)) for x=1:w]
    gh = [exp(-(x - mh)^2/(2*σ^2)) for x=1:h]
    gwn = gw / sum(gw)
    ghn = gh / sum(gh)
    ghn * gwn'
end

convolve_image(image, kernel) = irfft(rfft(image) .* rfft(kernel), size(image, 1))

function loss(points, log_bandwidth, log_weight)
    model_image = model(points, log_bandwidth, log_weight)
    sum((model_image .- target_image).^2) + sum(points.^2)
end

logrange(s, e, n) = round.(Int, exp.(range(log(s), log(e), n)))

function langevin!(points, log_bandwidth, log_weight, eps, n, update_bandwidth=true, update_weight=true, eps_after_init=eps; n_init=n, n_logs_init=15)
    logs_init = logrange(1, n_init, n_logs_init)
    log_every = false
    logstep = 1
    for i in 1:n
        l, grads = Zygote.withgradient(loss, points, log_bandwidth, log_weight)
        points .+= sqrt(eps) .* randn(Float32, size(points)) .- eps .* 0.5f0 .* grads[1]
        if update_bandwidth
            log_bandwidth += sqrt(eps) * randn(Float32) - eps * 0.5f0 * grads[2]
        end
        if update_weight
            log_weight += sqrt(eps) * randn(Float32) - eps * 0.5f0 * grads[3]
        end

        if i == n_init
            log_every = true
        end
        if log_every || (i in logs_init)
            println("iteration $logstep, $i: loss = $l, bandwidth = $(exp(log_bandwidth)), weight = $(exp(log_weight))")
            save("image_$logstep.png", Gray.(clamp01.(model(points, log_bandwidth, log_weight))))
            logstep += 1
        end
    end
    save("image_final.png", Gray.(clamp01.(model(points, log_bandwidth, log_weight))))
    points, log_bandwidth
end

isinteractive() || langevin!(points, log(0.5f0), 0f0, 5f-6, 6_030, false, true, 5f-4; n_init=6_000)
