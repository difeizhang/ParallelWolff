GC.gc()
__precompile__()

using BenchmarkTool


module execute
using Statistics
using DataStructures
using StaticArrays
using Random

const βc = log(1 + √2)/2
const beta_crit = βc

include("ParallelWolff.jl" ) # parallel Wolff algorithm # ParallelIsingWolff
end

function energy(lattice::AbstractArray{Int64})
    # calculate energy
    energy = 0
    for i in 1:size(lattice, 1)
        for j in 1:size(lattice, 2)
            energy -= lattice[i, j] * (lattice[mod1(i+1, size(lattice, 1)), j] + lattice[i, mod1(j+1, size(lattice, 2))])
        end
    end
    return energy
end

#    for β in 0.1:0.1:1.0
lattice_length = 20
lattice_shape = (lattice_length, lattice_length, lattice_length, lattice_length)
num = 10  # number of samples to generate
save_every = 1  # Wolff steps between saved samples
therm = 10 # thermalization steps
energy_means = zeros(10)
β = log(1 + √2)/2
execute.wolffsample(lattice_shape, β, num, save_every, therm);