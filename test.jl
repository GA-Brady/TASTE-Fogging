using LinearAlgebra
using AbstractFFTs
using FFTW
using Plots

function main()
    test = log2(17)
    test % 1 == 0 ? print("Integer") : print("Float")
end


if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end