# Wolff Algorithm for Ising Model in Julia (with StaticArrays) 

#### Basic Wolff algorithm #####
const LatticeArray = SizedArray{S,Int,N,N,TData} where {TData<:AbstractArray{Int,N}} where {N,S} # N-dimensional lattice of size S with elements of type TData (default Int64) 

function wolffstep!(rng, lattice::LatticeArray{N}, p_add, directions, lattice_indices, queue, shape) where {N}
    start = rand(rng, lattice_indices)
    store = []
    push!(store, SVector(Tuple(start)...))
    push!(store, SVector(Tuple(start)...))
    push!(store, SVector(Tuple(start)...))
    push!(store, SVector(Tuple(start)...))
    enqueue!(queue, SVector(Tuple(start)...))
    clustersign = @inbounds lattice[start]
    newsign = -clustersign
    @inbounds lattice[start] = newsign

    current_boundary = []
    next_boundary = []
    push!(current_boundary, SVector(Tuple(start)...))

    while !isempty(queue) # while queue is not empty
        empty!(queue)
        boundary_length = length(current_boundary)
        for direction in directions
            #Threads.@threads for boundary_element in current_boundary
            if boundary_length < 100
                for i in eachindex(current_boundary)
                    #next = @. mod(current_boundary[i] + direction - 1, shape) + 1 # periodic boundary conditions (1-based indexing)
                    @inbounds store[Threads.threadid()] = @. mod(current_boundary[i] + direction - 1, shape) + 1
                    @inbounds next_ci = CartesianIndex{N}(store[Threads.threadid()]...) # convert to CartesianIndex  
                    if @inbounds(lattice[next_ci]) == clustersign
                        if rand(rng) < p_add
                            enqueue!(queue, store[Threads.threadid()]) # push next point onto queue
                            @inbounds lattice[next_ci] = newsign
                            push!(next_boundary, store[Threads.threadid()])
                        end
                    end
                end
            else
                Threads.@threads for i in eachindex(current_boundary)
                    @inbounds store[Threads.threadid()] = @. mod(current_boundary[i] + direction - 1, shape) + 1 # periodic boundary conditions (1-based indexing)
                    if @inbounds(lattice[CartesianIndex{N}(store[Threads.threadid()]...)]) == clustersign
                        if rand(rng) < p_add
                            enqueue!(queue, store[Threads.threadid()]) # push next point onto queue
                            @inbounds lattice[CartesianIndex{N}(store[Threads.threadid()]...)] = newsign
                            push!(next_boundary, store[Threads.threadid()])
                        end
                    end
                end
            end
        end
        #print("next_boundary = ", next_boundary, "\n")
        #pop!(current_boundary)
        current_boundary = copy(next_boundary)
        next_boundary = empty!(next_boundary)
    end
    nothing
end


function savesample!(i, samples::AbstractArray{Bool}, lattice, lattice_indices) # save sample to samples array (Bool)
    @inbounds @. samples[i, lattice_indices] = (lattice == 1) # convert to Boolean and save
end

function savesample!(i, samples::AbstractArray{Int}, lattice, lattice_indices) # save sample to samples array (Int)
    @inbounds @. samples[i, lattice_indices] = lattice # save sample
end


function wolffsample!( # generate samples using Wolff algorithm (preallocated samples array)
    samples::AbstractArray{T}, # preallocated samples array (T = Int or Bool)
    rng::AbstractRNG, # random number generator (e.g. MersenneTwister)
    lattice::AbstractArray{Int, N}, # initial lattice (Int)
    β::Real, # inverse temperature
    nsamples, # number of samples to generate
    keep_every = 100, # number of Wolff steps between saved samples
    ntherm = 0, # number of thermalization steps
) where {N, T <: Union{Int, Bool}} # N dimensions, T = Int or Bool (Int = -1 or 1)
    lattice = LatticeArray{N, Tuple{size(lattice)...}}(lattice) # convert to SizedArray
    p_add = 1 - exp(-2β) # probability of adding a spin to the cluster
    lat_ind = CartesianIndices(lattice) # indices of lattice  (CartesianIndex)
    # can go up and down in each of the N dimensions
    directions = SVector{2N}([
        dir * SVector{N}(setindex!(zeros(Int64, N), 1, i))
        for i in 1:N for dir in [1, -1]])
    # queue is allocated once then reused in the wolff step
    queue = Queue{SVector{N, Int64}}()
    # shape of lattice
    shape = SVector{N}(size(lattice)...) 

    # thermalization
    for i in 1:ntherm 
        @time wolffstep!(rng, lattice, p_add, directions, lat_ind, queue, shape)
        #GC.gc() 
    end
    #savesample!(1, samples, lattice, lat_ind) # save first sample 
    for i in 2:nsamples # generate samples 
        for _ in 1:keep_every # keep_every steps between samples 
            @time wolffstep!(rng, lattice, p_add, directions, lat_ind, queue, shape)
            #GC.gc() 
        end 
        #savesample!(i, samples, lattice, lat_ind) # save sample
    end
    
    return samples # return samples
end

# start with Boolean lattice
function wolffsample!(samples::AbstractArray, rng::AbstractRNG, lattice::AbstractArray{Bool}, β, nsamples, keep_every, ntherm)
    return wolffsample!(samples, rng, (s -> s ? 1 : -1).(lattice), β, nsamples, keep_every, ntherm)
end

# start with random lattice
function wolffsample!(samples::AbstractArray, rng::AbstractRNG, lattice_shape::Tuple, β, nsamples, keep_every, ntherm)
    lattice = rand(rng, (-1, 1), lattice_shape)
    return wolffsample!(samples, rng, lattice, β, nsamples, keep_every, ntherm)
end

# with random seed
function wolffsample!(samples::AbstractArray, seed::Int, lattice::Union{Tuple,AbstractArray}, β, nsamples, keep_every, ntherm)
    rng = Xoshiro(seed)
    return wolffsample!(samples, rng, lattice, β, nsamples, keep_every, ntherm)
end

# with default rng
function wolffsample!(samples::AbstractArray, lattice::Union{Tuple,AbstractArray}, β, nsamples, keep_every, ntherm)
    rng = Random.default_rng()
    return wolffsample!(samples, rng, lattice, β, nsamples, keep_every, ntherm)
end

# without pre-allocated sample array (shape)
function wolffsample(rng::Union{Int,AbstractRNG}, lattice_shape::NTuple{N,Int}, β, nsamples, keep_every, ntherm) where {N}
    samples = Array{Int,N + 1}(undef, nsamples, lattice_shape...) 
    wolffsample!(samples, rng, lattice_shape, β, nsamples, keep_every, ntherm)
    return samples
end

# without pre-allocated sample array (array)
function wolffsample(rng::Union{Int,AbstractRNG}, lattice::AbstractArray{T,N}, β, nsamples, keep_every, ntherm) where {T,N}
    samples = Array{Int,N + 1}(undef, nsamples, size(lattice)...)
    wolffsample!(samples, rng, lattice, β, nsamples, keep_every, ntherm)
    return samples
end

# without pre-allocated sample array & without rng
function wolffsample(lattice::Union{Tuple,AbstractArray}, β, nsamples, keep_every, ntherm) where {N}
    rng = Random.default_rng()
    return wolffsample(rng, lattice, β, nsamples, keep_every, ntherm)
end