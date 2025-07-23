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

function main()
    η = .2
    n = 1000 # microns (rescaling this number rescales the gridspec)
    test_σb = 30 # microns

    pattern_mask = ones(Float64, n, n) # assuming that sigma forward < partician
    g_matrix = gauss_matrix(test_σb, 1)# assuming that delta x = 1 micron
    m, _ = size(g_matrix)
    # throw in a try-catch block to catch if kernel is bigger than gridspec

    # n = n % 2 == 0 ? n : n - 1 # reduces the size by 1 if the grid is odd to properly center the data
    pad = nextpow(2, n + m - 1)

    if pad == n
        pad = nextpow(2, n+1) # bumps up the padding if n = pad so that there is a border for the pattern kernel
    end

    println("Padding $n to $pad")

    pad_mask = grid_padder(pattern_mask, pad)
    pad_g = ifftshift(grid_padder(g_matrix, pad))

    patterned_g = real.((ifft(fft(pad_mask) .* fft(pad_g))))
    
    lower = round(Int, (pad-n) / 2) + 1
    upper = lower + n - 1
    
    initial_dose = zeros(Float64, n, n)
    initial_dose = (.5 .+ η) ./ (.5 .+ η .* (patterned_g[lower:upper, lower:upper]))

    test, corner_sample, center_sample = converge_dose_grid(pad, 20, lower, upper, η, grid_padder(initial_dose, pad), pad_g, initial_dose)

    p1 = heatmap(test)
    p2 = plot(corner_sample)
    p3 = plot(center_sample)
    savefig(p1, "figures/fft_ifftshifted_heatmap.png")
    savefig(p2, "figures/corner_sample.png")
    savefig(p3, "figures/center_sample.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end