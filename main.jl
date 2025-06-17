using Distributions
using DataFrames
using SpecialFunctions
using LinearAlgebra

function summed_gauss(σ::Float64, x::AbstractVector, grid::AbstractVector)
    if σ <= 0
        println("Invalid sigma value")
        return nothing 
    end

    A = 1 / sqrt(2*pi) * (1 / σ)   
    contributions = 0

    x_center = x[1]
    y_center = x[2]
    
    x_len = grid[1]
    y_len = grid[2]

    for xi in 1:x_len
        ΔX = x_center - xi
        ΔX_squared = ΔX^2

        for yi in 1:y_len
            ΔY = y_center - yi
            ΔY_squared = ΔY^2

            r_squared = ΔX_squared + ΔY_squared
            r = sqrt(r_squared)

            if r <= (3 * σ)
                backscatter = A * ℯ ^ (-r_squared / (2*σ^2))
                volumetric_backscatter = backscatter * 2 * pi * ΔX * ΔY
                contributions += volumetric_backscatter
            end
        end
    end
    return contributions
end
function normalized_proximity(η::Float64,σ::Float64,x::AbstractVector,grid::AbstractVector)
    #=
    in: 
        η - ratio of deposited energy in backscattered to forward scattered electrons
        σ - backscattering electron range
        x - beam position in the resist
        grid - describes the dimensions of the grid
    out:
        corrected normalized proximity dose at given beam position
    =#
    
end

function main()
    global primary_resolution = 100
    global secondary_resulution = 10 # temporary values to give physical meaning later

    n = 100
    # Iterating through each sample point in the array
    print(summed_gauss(2.0,[0.0,0.0],[10,10]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end