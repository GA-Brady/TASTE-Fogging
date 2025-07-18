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

function main_2()
    test_σ_back = 10
    test_ΔX = 1

    n = round(Int, (3*test_σ_back / test_ΔX))
    n = n % 2 == 0 ? n : n - 1 # reduces the size by 1 if the grid is even to properly center the data
    pad = nextpow(2, n) # calculating the next power of 2 to speed up FFT

    dl_index = round(Int, (pad-n) / 2 ) + 1
    dr_index = dl_index + n - 1

    g = gauss_matrix(test_σ_back, test_ΔX)
    mask = ones(Float64, n,n)

    pad_mask = collect(PaddedView(0, mask, (1:pad, 1:pad), (dl_index:dr_index, dl_index:dr_index)))
    pad_g = collect(PaddedView(0, g, (1:pad, 1:pad), (dl_index:dr_index, dl_index:dr_index)))

    patterned_g = real.(ifft(fft(pad_mask) .* fft(pad_g)))
    
    p1 = heatmap(patterned_g)
    savefig(p1, "figures/fft_heatmap.png")

end

function grid_padder(a::Matrix, pad::Integer, value::Union{Float64, Integer} = 0)
    #=
        padding function for matrices
    =#
    n, __ = size(a)

    lower = round(Int, (pad-n) / 2) +1
    upper = lower + n - 1

    return collect(PaddedView(value, a, (1:pad, 1:pad), (lower:upper, lower:upper)))
end

function main()
    n = 13 # microns (rescaling this number rescales the gridspec)
    test_σb = 4 # microns
    
    # throw in a try-catch block to catch if kernel is bigger than gridspec

    n = n % 2 == 0 ? n : n - 1 # reduces the size by 1 if the grid is odd to properly center the data
    pad = nextpow(2, n)

    pattern_mask = ones(Float64, n, n) # assuming that sigma forward < partician
    dose_grid = ones(Float64, n, n)
    g_matrix = gauss_matrix(test_σb, 1) # assuming that delta x = 1 micron

    pad_mask = grid_padder(pattern_mask, pad)
    pad_g = grid_padder(g_matrix, pad)

    patterned_g = real.(ifft(fft(pad_mask) .* fft(pad_g)))
    p1 = heatmap(patterned_g)
    savefig(p1, "figures/fft_heatmap.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end