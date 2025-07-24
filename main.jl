using DataFrames
using LinearAlgebra
using AbstractFFTs
using FFTW
using PaddedViews
using Plots

function gauss_2d(σ::Union{Float64, Integer}, center::Vector, position::Vector)
    Δr_squared = sum((position .- center) .^2)
    exponent = - 0.5 * Δr_squared / (σ^2)
    return 1 / (2*pi*σ) * exp(exponent)
end

function gauss_matrix(σ::Union{Float64, Integer}, ΔX ::Union{Integer, Float64})
    #=
    generates diagonal matrix from gaussian view window
    in:
        σ - standard deviation
        ΔX - side length of grid cell
    =#
    partician = 3 * σ / ΔX
    n = round(Int, partician)
    n = n % 2 == 0 ? n : n - 1 # reduces the size by 1 if the grid is odd to properly center the data

    g_matrix = zeros(Float64, n,n)

    for i in 1:n
        for j in 1:n
            g_matrix[i,j] = gauss_2d(σ, [(n+1)/2, (n+1)/2], [i, j])
        end
    end

    correction = sum(g_matrix)
    g_matrix = g_matrix ./ correction

    return g_matrix
end

function grid_padder(a::Matrix, pad::Integer, value::Union{Float64, Integer} = 0)
    #=
        padding function for matrices
    =#
    n, __ = size(a) # assumes all input matrices are squares

    lower = floor(Int, (pad-n) / 2) + 1
    upper = lower + n - 1

    return collect(PaddedView(value, a, (1:pad, 1:pad), (lower:upper, lower:upper)))
end

function dose_grid_step(lower::Integer, upper::Integer, η::Union{Float64, Integer}, dose_grid::Matrix, kernel::Matrix, modified_pattern::Matrix)
    dose_grid_roi = dose_grid[lower:upper, lower:upper]
    
    conv = real.((ifft(fft(dose_grid) .* fft(kernel))))
    conv_roi = conv[lower:upper, lower:upper] # assume kernel is already ifft shfited

    A = 1 / (.5 + η)
    ϵ = zeros(Float64, size(modified_pattern))
    ϵ = A .* (dose_grid_roi ./ 2 .+ η .* conv_roi) .- 1
    dose_correction = .- ϵ .* modified_pattern .+ dose_grid_roi
    
    return dose_correction
end

function converge_dose_grid(pad, iter, lower, upper, η, dose_grid, kernel, modified_pattern)
    corner_sample = []
    center_sample = []

    n, _ = size(modified_pattern)
    center = Int(floor(n/2) + 1)
    
    for i in 1:iter
        println("Iterating step $i / $iter")
        padded_dose = grid_padder(dose_grid, pad)
        dose_grid = dose_grid_step(lower, upper, η, padded_dose, kernel, modified_pattern)

        push!(corner_sample, dose_grid[2,2])
        push!(center_sample, dose_grid[center,center])
    end

    return dose_grid, corner_sample, center_sample
end

function fogging_grid_step(lower, upper, ζ, fogging_grid, fogging_kernel, pattern, initial_fogging)
    fogging_grid_roi = fogging_grid[lower:upper, lower:upper]

    conv = real.(ifft(fft(fogging_grid .* pattern) .* fft(fogging_kernel)))
    conv_roi = conv[lower:upper, lower:upper]

    ϵ = zeros(Float64, size(fogging_grid_roi))
    ϵ = fogging_grid_roi .+ ζ .* conv_roi .- 1

    f = .- ϵ + initial_fogging
    F = fogging_grid_roi .+ f

    return F
end

function converge_fogging_grid(pad, iter, lower, upper, ζ, fogging_grid, fogging_kernel, pattern, initial_fogging)
    corner_sample = []
    center_sample = []

    n, _ = size(initial_fogging)
    center = Int(floor(n / 2 ) + 1)

    for i in 1:iter
        println("Iterating step $i / $iter")
        padded_fogging = grid_padder(fogging_grid, pad)
        fogging_grid = fogging_grid_step(lower, upper, ζ,  padded_fogging, fogging_kernel, pattern, initial_fogging)
        
        push!(corner_sample, fogging_grid[2,2])
        push!(center_sample, fogging_grid[center, center])
    end

    return fogging_grid, corner_sample, center_sample
end

function main()
    η = .2
    ζ = .1
    n = 1000 # microns (rescaling this number rescales the gridspec)
    test_σb = 30 # microns
    test_σf = 300 # microns

    pattern_mask = ones(Float64, n, n) # assuming that sigma forward < partician
    backscatter_kernel = gauss_matrix(test_σb, 1)# assuming that delta x = 1 micron
    m, _ = size(backscatter_kernel)
    # throw in a try-catch block to catch if kernel is bigger than gridspec

    # n = n % 2 == 0 ? n : n - 1 # reduces the size by 1 if the grid is odd to properly center the data
    pad = nextpow(2, n + m - 1)

    println("Padding $n to $pad")

    pad_mask = grid_padder(pattern_mask, pad)
    pad_backscatter_kernel = ifftshift(grid_padder(backscatter_kernel, pad))

    patterned_backscatter_kernel = real.((ifft(fft(pad_mask) .* fft(pad_backscatter_kernel))))
    
    lower = round(Int, (pad-n) / 2) + 1
    upper = lower + n - 1
    
    initial_dose = zeros(Float64, n, n)
    initial_dose = (.5 .+ η) ./ (.5 .+ η .* (patterned_backscatter_kernel[lower:upper, lower:upper]))

    proximity_correction, corner_sample, center_sample = converge_dose_grid(pad, 5, lower, upper, η, grid_padder(initial_dose, pad), pad_backscatter_kernel, initial_dose)

    p1 = heatmap(proximity_correction)
    savefig(p1, "figures/proximity_map.png")
    p2 = plot(corner_sample)
    p3 = plot(center_sample)
    p4 = plot(p2, p3)
    savefig(p4, "figures/convergence_test.png")

    modified_pattern_density = .75 .* proximity_correction # only when P and F spacing the same - implement function later

    fogging_kernel = gauss_matrix(test_σf, 1)
    mf, _ = size(fogging_kernel) # gauss matrix returns a sqaure
    
    fog_padding = nextpow(2, n + mf -1)
    println("Padding $n to $fog_padding for fogging convolution:")

    pad_fogging_kernel = ifftshift(grid_padder(fogging_kernel, fog_padding))
    pad_modified_pattern_density = grid_padder(modified_pattern_density, fog_padding)

    fog_lower = Int(floor((fog_padding - n) / 2) + 1)
    fog_upper = fog_lower + n - 1

    convolved_pattern_density = real.(ifft(fft(pad_fogging_kernel) .* fft(pad_modified_pattern_density)))[fog_lower:fog_upper, fog_lower:fog_upper]
    initial_fogging = 1 ./ (1 .+ ζ .* convolved_pattern_density)

    fogging_correction, fogging_corner, fogging_center = converge_fogging_grid(fog_padding, 20, fog_lower, fog_upper, ζ, initial_fogging, pad_fogging_kernel, pad_modified_pattern_density, initial_fogging)

    initial_fogging_heatmap = heatmap(fogging_correction)
    savefig(initial_fogging_heatmap, "figures/initial_fogging.png")

    p1 = plot(fogging_corner)
    p2 = plot(fogging_center)
    savefig(plot(p1, p2), "figures/fogging_convergence.png")

    dose_correction = fogging_correction .* proximity_correction

    savefig(heatmap(dose_correction), "figures/dose_correction.png")

end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end