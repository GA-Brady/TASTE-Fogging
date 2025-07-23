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

function main()
    η = .1
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
    pad_g = grid_padder(g_matrix, pad)

    # display(pad_g)

    patterned_g = real.((ifft(fft(pad_mask) .* fft(ifftshift(pad_g)))))

    lower = round(Int, (pad-n) / 2) + 1
    upper = lower + n - 1

    # dose_grid_it = converge_dose_grid(pad, lower, upper, η, padded_dose_grid, pad_g, dose_grid)
    #=
    println("Padded Mask:")
    display(pad_mask)
    println("Pad G:")
    display(pad_g)
    println("FFT Convolution output:")
    display(patterned_g)
    =#
    p1 = heatmap(patterned_g[lower:upper, lower:upper])
    savefig(p1, "figures/fft_ifftshifted_heatmap.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end