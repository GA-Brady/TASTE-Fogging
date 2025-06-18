using Distributions
using DataFrames
using SpecialFunctions
using LinearAlgebra
using Plots

function volumetric_gauss(σ::Union{Float64, Integer}, x::AbstractVector, y::AbstractVector)
    #=
    function which returns the 2d volumetric approximation of the multivariate gauss distribution
    in:
        σ - standard deviation of the gaussian approximation
        x - beam position in resist
        y - imaginary beam position
    out:
        gauss approximation
    =#

    ΔX = abs(x[1] - y[1])
    ΔX_squared = ΔX^2

    ΔY = abs(x[2] - y[2])
    ΔY_squared = ΔY^2

    r_squared = ΔX_squared + ΔY_squared
    r = sqrt(r_squared)

    if r <= (3 * σ)
        A = 1 / (sqrt(2*pi) * σ) 
        scatter = A * ℯ ^ (- r_squared / (2*σ^2))
        volumetric = scatter * 2 * pi * r * ΔX * ΔY
        return volumetric
    end

    return 0.0 
end

function summed_gauss(σ::Union{Float64, Integer}, x::AbstractVector, etched_grid::Matrix)
    #=
    in:
        σ - standard deviation of gaussian approximation
        x - beam position in resist
        grid - bitmap which masks the etched and non etched areas, assumes grid is populated
    =#
    if σ <= 0
        println("Invalid sigma value")
        return nothing 
    end

    x_len, y_len = size(etched_grid)
    t = 0

    for xi in 1:x_len
        for yi in 1:y_len
            if etched_grid[xi, yi] > 0
                s = volumetric_gauss(σ, x, [xi, yi])
                t += s
            end
        end
    end
    return t
end

function instantiate_proximity_dose_grid(σ::Union{Float64, Integer}, η::Union{Float64, Integer}, etched_grid::Matrix)
    #=
    Creates the base case dosing grid
    =#
    n, m = size(etched_grid)
    prox_dose_grid = zeros(Float64, n, m)

    for xi in 1:n
        for yi in 1:m
            prox_dose_grid[xi, yi] = (.5 + η) / (.5 + η * summed_gauss(σ, [xi, yi], etched_grid))
        end
    end

    return prox_dose_grid
end

function iterate_dose_grid_element(σ::Union{Float64, Integer}, η::Union{Float64, Integer}, x::AbstractVector, initial_dose_grid::Matrix, dose_grid::Matrix)

    #=
    Iterates an element of the dosing grid 
    =#
    n, m = size(initial_dose_grid)
    t = 0

    for xi in 1:n
        for yi in 1:m
            s = dose_grid[xi, yi] * volumetric_gauss(σ, x, [xi, yi])
            t += s
        end
    end

    ϵ = (1 / (0.5 + η)) * (( .5 * dose_grid[x[1], x[2]]) + η * t ) - 1
    d = - ϵ * initial_dose_grid[x[1], x[2]]
    D = dose_grid[x[1], x[2]] + d

    return D
end

function iterate_dose_grid(σ::Union{Float64, Integer}, η::Union{Float64, Integer}, initial_dose_grid::Matrix, dose_grid::Matrix)
    #=
    Iterates the entire dosing grid
    =#
    n, m = size(initial_dose_grid)
    new_dose_grid = zeros(Float64, n, m)

    for xi in 1:n
        for yi in 1:m
            new_dose_grid[xi, yi] = iterate_dose_grid_element(σ, η, [xi, yi], initial_dose_grid, dose_grid)
        end
    end

    return new_dose_grid
end

function resolve_dose_grid(σ::Union{Float64, Integer}, η::Union{Float64, Integer}, res::Integer, etched_grid::Matrix)
    #=
    Insert Docstring here
    =#
    D1_grid = instantiate_proximity_dose_grid(σ, η, etched_grid)

    dose_grid = iterate_dose_grid(σ, η, D1_grid, D1_grid)

    for i in 1:res
        if (i+1) % 5 == 1 
            println("Iteration $i / $res")
        end

        s = iterate_dose_grid(σ, η, D1_grid, dose_grid)
        dose_grid = s
    end

    return dose_grid
end

function main()
    global primary_resolution = 100
    global secondary_resulution = 10 # temporary values to give physical meaning later

    n = 100
    test_matrix = Matrix{Int}(I, n, n) # using itentity matrix as test matrix
    converged_dose = resolve_dose_grid(10, .80, 10, test_matrix)

    heatmap_1 = heatmap(converged_dose)
    savefig(heatmap_1, "figures/test_heatmap")
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end