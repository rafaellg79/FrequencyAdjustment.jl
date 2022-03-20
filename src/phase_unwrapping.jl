using Printf
using Combinatorics
using StatsBase: psnr

"""
    detectflip(w1::Wave, w2::Wave)

Detect if a sign flip occurred between waves `w1` and `w2` as described
in Section 4.1.2 of Frequency Adjustment paper.
"""
function detectflip(w1::Wave, w2::Wave)
    # Let p and q be the frequencies of w1 and w2, respectively.
    p = w1.freqs
    q = w2.freqs
    # Let distance be the L2 distance function between two points
    distance = (p,q) -> norm(p .- q)
    # Compute the distance between p and q
    dist_pos = distance(p, q)
    # And compute the distance between p and -q
    dist_neg = distance(p, .-q)
    # Return that a flip occurred if the flipped vector (-q)
    # preserves the frequency direction better than the
    # original vector (q)
    return dist_neg < dist_pos
end

"""
    alignmentmeasure(w1::Wave{T, 2}, w2::Wave{T, 2}, ind1::CartesianIndex{2}, ind2::CartesianIndex{2})

Compute the alignment between waves `w1` and `w2` at indices `ind1` and `ind2`.
Uses the measure ``μ`` described in Section 4.3.1 of the Spectral Remapping paper
multiplied by the frequency-vector orientation similarity measure described in
Equation 31 of the Frequency Adjustment paper.
"""
alignmentmeasure(w1, w2, ind1, ind2) = error("Error: Trying to measure alignment of $(w1) with $(w2) at $(ind1) and $(ind2)") # Fallback when any of the waves do not match type

function alignmentmeasure(w1::Wave{T, 2}, w2::Wave{T, 2}, ind1::CartesianIndex{2}, ind2::CartesianIndex{2}) where T
    # Point halfway between the windows, where the phase of the waves
    # should match.
    ym, xm = T.((ind1.I .+ ind2.I)./2)

    # 3x3 neighborhood around (xm,ym)
    yms = ym .+ (-1:1)
    xms = xm .+ (-1:1)'

    # Compute the alignment measure μ between the two waves.
    vals1 = @. w1.amplitude * cos(2T(π) * (w1.freqs[2]*xms + w1.freqs[1]*yms + w1.phase))
    vals2 = @. w2.amplitude * cos(2T(π) * (w2.freqs[2]*xms + w2.freqs[1]*yms + w2.phase))
    δ = norm(vals1 - vals2) / min(w1.amplitude, w2.amplitude)
    λ = T(0.5)
    μ = δ > 3λ ? zero(T) : exp(-δ^2 / λ^2)
    
    # Detect if a flip occurs and store an unary sign operator into flipsign
    flipsign = detectflip(w1, w2) ? (-) : (+)
    ab1 = w1.freqs
    # Broadcast sign operator into the elements of w2.freqs
    ab2 = w2.freqs .|> flipsign
    # Multiply μ by the frequency-vector orientation similarity measure
    μ  *= dot(ab1,ab2) / norm(ab1) / norm(ab2)

    return μ
end

"""
    alignmentmeasure(w1::Wave{T, 3}, w2::Wave{T, 3}, ind1::CartesianIndex{3}, ind2::CartesianIndex{3})

Compute the alignment between waves `w1` and `w2` at indices `ind1` and `ind2`.
Uses the measure ``μ`` described in Section 4.3.1 of the Spectral Remapping paper
multiplied by the frequency-vector orientation similarity measure described in
Equation 31 of the Frequency Adjustment paper.
"""
function alignmentmeasure(w1::Wave{T, 3}, w2::Wave{T, 3}, ind1::CartesianIndex{3}, ind2::CartesianIndex{3}) where T
    # Point halfway between the windows, where the phase of the waves
    # should match.
    ym, xm, zm = T.((ind1.I .+ ind2.I)./2)

    # 3x3x3 neighborhood around (xm,ym)
    yms = ym .+ (-1:1)
    xms = xm .+ (-1:1)'
    zms = reshape(zm .+ (-1:1), (1, 1, 3))

    # Compute the alignment measure μ between the two waves.
    vals1 = @. w1.amplitude * cos(2T(π) * (w1.freqs[2]*xms + w1.freqs[1]*yms + w1.freqs[3]*zms + w1.phase))
    vals2 = @. w2.amplitude * cos(2T(π) * (w2.freqs[2]*xms + w2.freqs[1]*yms + w2.freqs[3]*zms + w2.phase))
    δ = norm(vals1 - vals2) / min(w1.amplitude, w2.amplitude)
    λ = T(0.5)
    μ = δ > 3λ ? zero(T) : exp(-δ^2 / λ^2)

    # Detect if a flip occurs and store an unary sign operator into flipsign
    flipsign = detectflip(w1, w2) ? (-) : (+)
    ab1 = w1.freqs
    # Broadcast sign operator into the elements of w2.freqs
    ab2 = w2.freqs .|> flipsign
    # Multiply μ by the frequency-vector orientation similarity measure
    μ  *= dot(ab1,ab2) / norm(ab1) / norm(ab2)

    return μ
end

"""
    alignmentmeasure(w1::Wave{T, N}, w2::Wave{T, N}, ind1::CartesianIndex{N}, ind2::CartesianIndex{N})

Compute the alignment between waves `w1` and `w2` at indices `ind1` and `ind2`.
Uses the measure ``μ`` described in Section 4.3.1 of the Spectral Remapping paper
multiplied by the frequency-vector orientation similarity measure described in
Equation 31 of the Frequency Adjustment paper.
"""
function alignmentmeasure(w1::Wave{T,N}, w2::Wave{T,N}, ind1::CartesianIndex{N}, ind2::CartesianIndex{N}) where T where N
    # Point halfway between the windows, where the phase of the waves
    # should match.
    m = T.((ind1.I .+ ind2.I)./2)

    # 3x3...x3 neighborhood around (xm,ym)
    pos = zeros(T, N)
    vals = zeros(T, ntuple(n -> 3, Val(N)))
    @inbounds for i in 1:length(vals)
        phase1 = zero(T)
        phase2 = zero(T)
        # Compute the phase at linear index i of planar waves w1 and w2 by integrating the frequency
        for n in 1:N
            dn = ((i-1) ÷ 3^(n-1) % 3)-1 + m[n]
            phase1 += w1.freqs[n] * dn
            phase2 += w2.freqs[n] * dn
        end
        # Compute the wave value at linear index i
        vals[i] = w1.amplitude * cos(2T(π) * (phase1 + w1.phase)) -
                  w2.amplitude * cos(2T(π) * (phase2 + w2.phase))
    end

    # Compute the alignment measure μ between the two waves.
    δ = norm(vals) / min(w1.amplitude, w2.amplitude)
    λ = T(0.5)
    μ = δ > 3λ ? zero(T) : exp(-δ^2 / λ^2)

    # Detect if a flip occurs and store an unary sign operator into flipsign
    flipsign = detectflip(w1, w2) ? (-) : (+)
    ab1 = w1.freqs
    # Broadcast sign operator into the elements of w2.freqs
    ab2 = w2.freqs .|> flipsign
    # Multiply μ by the frequency-vector orientation similarity measure
    μ  *= dot(ab1,ab2) / norm(ab1) / norm(ab2)

    return μ
end

using Base: @propagate_inbounds

"""
    CompactLinearIndices(index, lastindex)

An object for linear indexing a matrix of arrays.
Array elements of the matrix form the major indexing dimension
followed by the usual Julia column indexing order.
"""
struct CompactLinearIndices
    index::SpacedArray{Int}
    lastindex::Int
    function CompactLinearIndices(waves::SpacedArray{Vector{T}}) where T
        waves_per_pixel = parent(map(length, parent(waves)))
        waves_linear_counting = vec(waves_per_pixel)
        waves_linear_counting = cumsum(waves_linear_counting) .- waves_linear_counting .+ 1
        waves_linear_counting = reshape(waves_linear_counting, size(waves))
        index = SpacedArray(waves_linear_counting, axes(waves))
        lastindex = waves_linear_counting[end] + waves_per_pixel[end]
        return new(index, lastindex)
    end
end
"""
    numwaves(cli::CompactLinearIndices, pixelind::CartesianIndex)

Number of waves at pixelind of cli.
"""
@inline @propagate_inbounds function numwaves(cli::CompactLinearIndices, pixelind::CartesianIndex)
    i = LinearIndices(cli.index)[pixelind]
    get(cli.index, i+1, cli.lastindex) - cli.index[i]
end
@inline @propagate_inbounds function Base.checkbounds(::Type{Bool}, cli::CompactLinearIndices, pixelind::CartesianIndex, waveind)
    1 <= waveind <= numwaves(cli, pixelind)
end
@inline @propagate_inbounds function Base.checkbounds(cli::CompactLinearIndices, pixelind::CartesianIndex, waveind)
    if checkbounds(Bool, cli, pixelind, waveind) == false
        if waveind == 0
            error("BoundsError: attempt to access wave 0 at pixel $(pixelind.I).")
        else
            error("BoundsError: attempt to access wave $(waveind) at pixel $(pixelind.I) but it has only $(numwaves(cli,pixelind)) waves.")
        end
    end
end
@inline @propagate_inbounds function Base.getindex(cli::CompactLinearIndices, pixelind::CartesianIndex, waveind)
    @boundscheck checkbounds(cli, pixelind, waveind)
    return cli.index[pixelind] + waveind - 1
end
export CompactLinearIndices

include("wave_iterator.jl")

"""
    assemble_linear_system(waves::SpacedArray{Vector{Wave{T, N}}}, τ::Int)

Assemble the linear system of Equation 16 from the Frequency Adjustment paper
for waves sparsed in τ
"""
function assemble_linear_system(
  waves::SpacedArray{Vector{Wave{T, N}}}
, τ::Int
) where T where N
    # Maps a wave in the input to a linear system line
    wave_to_linearsystem_line = CompactLinearIndices(waves)
    
    # Estimating total memory necessary to pre-allocate
    # Number of pixels that possibly have a wave
    num_waves = length(waves)
    # Number of waves
    W = sum(length.(parent(waves)))
    # Estimate the maximum number of linear system lines for the derivative operators
    Mwaves = maximum(length.(parent(waves)))
    M = Mwaves^2 # Each wave can align with any other neighboring wave
    M = M * length(waves)
    # cs
    opcoord = LinearIndices((Mwaves, Mwaves, num_waves))
    
    # Backward-difference matrices described in Equation 14 of the Frequency Adjustment paper,
    # but for the system in Equation 17, i.e. weighted by waves alignment
    # Operators (lhs)
    D_Vs1 = zeros(T, M, N)
    D_Vs2 = zeros(T, M, N)
    D_Js1 = zeros(T, M, N)
    D_Js2 = zeros(T, M, N)
    # Values (rhs)
    f = Vector{T}[zeros(T, M) for i=1:N]
    
    # Used to enforce data terms
    largenumber = 1e6
    smallnumber = 1e-4
    # Compute valid locations in waves, i.e. locations with at least a wave.
    valid = map(!isempty, waves)
    count = Threads.Atomic{Int}(0)
    total = length(waves)
    P = abs(minimum(map(first, axes(waves)))) + 1
    @withprogress name="Assembling linear system" begin
        # Integer shift to compute derivative
        δ = zeros(Int, N)
        # for each pixel
        for (linearind, pixelind) in collect(enumerate(SpacedCartesianIndices(waves)))
            # if the pixel is valid
            if valid[pixelind]
                # Add axes forward diff lhs and rhs, if well defined
                for n = 1:N
                    δ .= 0
                    δ[n] = -1
                    # prevpixelind is pixelind shifted by δ
                    prevpixelind = SpacedArrays.spacedstep(waves, pixelind, δ...)
                    # If prevpixelind is in the padded image and is valid
                    if checkbounds(Bool, waves, prevpixelind) && valid[prevpixelind]
                        # For each pair of waves in pixelind and prevpixelind
                        for prevwaveind in 1:length(waves[prevpixelind]), waveind in 1:length(waves[pixelind])
                            # Get the pair of waves w1 and w2
                            w1 = waves[pixelind][waveind]
                            w2 = waves[prevpixelind][prevwaveind]
                            # Check if there is a flip in sign
                            flip = detectflip(w1, w2)
                            flipsign = flip ? (-) : (+)
                            # Compute the frequency partial derivative with respect to δ in pixelind
                            measureddiff::T = τ * 2T(pi) * mean([w1.freqs[n],flipsign(w2.freqs[n])])
                            # Evaluate the alignment between the waves
                            μ = alignmentmeasure(w1, w2, pixelind, prevpixelind)
                            # Get linear index of the pair of waves in the system
                            l = opcoord[waveind, prevwaveind, linearind]
                            
                            # Store coefficients into the linear system
                            # lhs
                            D_Js1[l, n] = wave_to_linearsystem_line[pixelind, waveind]
                            D_Vs1[l, n] = +1 * μ
                            D_Js2[l, n] = wave_to_linearsystem_line[prevpixelind, prevwaveind]
                            D_Vs2[l, n] = -1 * μ * (flip ? -1 : +1)
                            # rhs
                            f[n][l] = measureddiff * μ
                        end
                    end
                end
            end
            # Update progress
            count[] += 1 # Atomic increment
            if Threads.threadid() == 1
                @logprogress (count[] / total)
            end
        end
    end
    # Store differential matrices into an array
    D = SparseMatrixCSC{T}[]
    for n = 1:N
        nz1 = findall(!iszero, D_Js1[:, n])
        nz2 = findall(!iszero, D_Js2[:, n])
        push!(D, sparse(nz1, D_Js1[nz1, n], D_Vs1[nz1, n], M, W) +
             sparse(nz2, D_Js2[nz2, n], D_Vs2[nz2, n], M, W))
    end
    # Return tuple with differential matrices and frequencies
    return (D, f)
end

"""
    spblockdiagonal(Ms)

Given a vector with the differential matrices for each channel, combine the matrices into a single diagonal matrix
"""
function spblockdiagonal(Ms)
    m,n = sum.(zip(size.(Ms)...)) # Final size
    cumsiz = (0,0)
    i = 1
    M = Ms[1] # May be unecessary
    # Indices of non-zero elements to create a sparse matrix
    Is = Int[]
    Js = Int[]
    # Values of non-zero elements
    Vs = eltype(first(Ms))[]
    # For each channel system
    for (i,M) in enumerate(Ms)
        # Find non-zero indices
        rows, cols, vals = findnz(M)
        # Sum cumulative size of previous channels
        rows .+= cumsiz[1]
        cols .+= cumsiz[2]
        cumsiz = cumsiz .+ size(M)
        # Add to array of indices
        Is = [Is; rows]
        Js = [Js; cols]
        Vs = [Vs; vals]
    end
    # Create a sparse matrix with indices of non-zero elements and size (m, n)
    sparse(Is, Js, Vs, m, n)
end

"""
    cross_channel_constraints(channeldata)

Compute cross-channel constraints from an array with C named tuples with `waves` the array of detected waves, `D` and `f`the left and right side of the system of equations for each channel.
See Section 4.3 of the Frequency Adjustment paper for details on cross-channel constraints
"""
function cross_channel_constraints(channeldata)
    # Number of channels
    C = length(channeldata)
    # Given the diagonal matrix formed by the block matrices D, find the shift between blocks
    blockshifts = cumsum([0; [size(channeldata[i-1].D[1], 2) for i in 2:C]])
    waves = [channeldata[i].waves for i in 1:C]
    
    # Arrays for indexing non-zero elements
    cli = [CompactLinearIndices(channeldata[i].waves) for i in 1:C]
    Js1 = Int[]
    Js2 = Int[]
    Vs1 = Float64[]
    Vs2 = Float64[]
    bs  = Float64[]
    
    # For each pixelind in waves
    @progress "Adding cross-channel constraints" for pixelind in SpacedCartesianIndices(waves[1])
        # For each pair of channels
        for (chan1, chan2) in combinations(1:C, 2)
            # If there are waves in both channels at pixelind
            if checkbounds(Bool, waves[chan1], pixelind) && checkbounds(Bool, waves[chan2], pixelind)
                # Then for each pair of waves
                for waveind1 in 1:length(waves[chan1][pixelind]), waveind2 in 1:length(waves[chan2][pixelind])
                    w1 = waves[chan1][pixelind][waveind1]
                    w2 = waves[chan2][pixelind][waveind2]
                    # Compute possibility of being the same wave
                    flip = detectflip(w1, w2)
                    flipsign = flip ? (-) : (+)
                    ab1 = w1.freqs
                    ab2 = w2.freqs .|> flipsign
                    samedir  = dot(ab1,ab2) / norm(ab1) / norm(ab2)
                    freqdist = norm(norm(ab1) - norm(ab2)) * sqrt(2)
                    samefreq = exp(-freqdist^2 / 0.3^2)
                    μ = samedir * samefreq
                    # Add same-phase constraint to the linear system
                    j1 = cli[chan1][pixelind,waveind1] + blockshifts[chan1]
                    j2 = cli[chan2][pixelind,waveind2] + blockshifts[chan2]
                    v1 = +1 * μ
                    v2 = -1 * μ * (flip ? -1 : +1)
                    wrap(x) = x + 2π*fld(pi - x, 2π)
                    b  = μ * wrap(w1.phase - flipsign(w2.phase))
                    # Store indices into indexing arrays
                    push!(Js1, j1)
                    push!(Vs1, v1)
                    push!(Js2, j2)
                    push!(Vs2, v2)
                    push!(bs,  b)
                end
            end
        end
    end
    L = length(Js1)
    n = sum([size(channeldata[i].D[1], 2) for i in 1:C])
    CC = sparse(1:L, Js1, Vs1, L, n) +
         sparse(1:L, Js2, Vs2, L, n)
    return (CC, bs)
end

"""
    phaseunwrap(waves::AbstractArray{Vector{Wave{T, N}}, N}, τ; λ = 1e-6, freq_weights = ntuple((n) -> one(T), Val(N)))

Given a N-dimensional array of vector of N-dimensional waves sparsed by τ, integrate the waves frequencies to obtain the scalar potential field of unwrapped phases.
λ is the regularization term and freq_weights is a N-tuple that scales the frequencies.
"""
function phaseunwrap(waves::AbstractArray{Vector{Wave{T, N}}, N}, τ; λ = 1e-6, freq_weights = ntuple((n) -> one(T), Val(N))) where T<:AbstractFloat where N
    D, f = assemble_linear_system(waves, τ);
    
    # Given the system solve via leastsquares
    # Multiply each backward-difference matrix by its transpose and then sum all with the regularization term
    A = (sum((X) -> X'X, D) + λ*I)
    # Do the same for the right side of the system
    b = sum(adjoint.(D).*(freq_weights.*f))
    
    # Solve using Julia \ solver
    println("Solving the linear system.")
    return @time(T.(A \ b))
end

"""
    anisotropic_phaseunwrap(waves::AbstractArray{T, N}, τ; λ = 1e-6, freq_weights = ntuple((n) -> one(Float64), Val(N)))

Given a N-dimensional array of vector of N-dimensional waves sparsed by τ, integrate the waves frequencies for each axis to obtain a vector field of smooth phases.
λ is the regularization term.
"""
function anisotropic_phaseunwrap(waves::AbstractArray{Vector{Wave{T, N}}, N}, τ; λ = 1e-6) where T<:AbstractFloat where N
    D, f = assemble_linear_system(waves, τ);
    
    # Given the system solve via leastsquares
    # Multiply each backward-difference matrix by its transpose and then sum all with the regularization term
    A = (sum((X) -> X'X, D) + λ*I)
    u = map(1:N) do n
        # Do the same for the n-th right side of the system
        b = adjoint(D[n])*f[n]
        # Solve using Julia \ solver
        println("Solving the linear system for axis $(n) of $(N).")
        return @time(T.(A \ b))
    end
    
    return u
end

"""
    phaseunwrap_rgb(waves::NTuple{C,AbstractArray{Vector{Wave{T, N}}, N}}, τ; λ = 1e-6, β = 1e-3, freq_weights = ntuple((n) -> one(Float64), Val(N)))

Given a C-tuple of N-dimensional arrays of vector of N-dimensional waves sparsed by τ, jointly integrate the waves frequencies using the multichannel rgb optimization described in Section 4.3 of the Frequency Adjustment paper to obtain the scalar potential field of unwrapped phases.
λ is the regularization term, β scales the cross-channel phase constraints and freq_weights is a N-tuple that scales the frequencies.
"""
function phaseunwrap_rgb(waves :: NTuple{C,AbstractArray{Vector{Wave{T, N}}, N}}, τ; λ = 1e-6, β = 1e-3, freq_weights = ntuple((n) -> one(Float64), Val(N))) where C where T where N
    # Assemble and solve the multichannel linear system
    print("Phase unwrapping (RGB)... ")
    Ds = Vector{Vector{SparseMatrixCSC{T}}}(undef, C)
    fs = Vector{Vector{Vector{T}}}(undef, C)
    # Assemble the system for each channel
    for i in 1:C
        Ds[i], fs[i] = assemble_linear_system(waves[i], τ[i])
    end
    # Combine linear system data into a single array
    channeldata = [(waves=waves[i], D=Ds[i], f=fs[i]) for i in 1:C]
    # Compute cross-channel constraints
    CC,ccb = cross_channel_constraints(channeldata);
    D = [spblockdiagonal(getindex.(Ds, i)) for i = 1:N]
    f = [cat(getindex.(fs, i)...; dims=1) for i = 1:N]
    
    # Given the system solve via leastsquares
    # Multiply each backward-difference matrix by its transpose and then sum all with the cross-channel constraints and the regularization term
    A = (sum((X) -> X'X, D) + β*CC'CC + λ*I)
    # Do the same for the right side of the system
    b = sum(adjoint.(D).*(freq_weights.*f));
    # Solve
    print("Solving the linear system.")
    u = @time(A \ b)
    # Propagate the computed phase values to the FrequencyAdjuster of each channel
    cumsiz = 0
    u_all = ntuple(C) do channel
        n = size(channeldata[channel].D[1], 2)
        cumsiz_old = cumsiz
        cumsiz += n
        u[(1:n) .+ cumsiz_old]
    end
    GC.gc(true)
    println("done.")
    return u_all
end

"""
    anisotropic_phaseunwrap_rgb(waves :: NTuple{C,AbstractArray{Vector{Wave{T, N}}, N}}, τ; λ = 1e-6, β = 1e-3)

Given a C-tuple of N-dimensional arrays of vector of N-dimensional waves sparsed by τ, jointly integrate the waves frequencies for each axis using the multichannel rgb optimization described in Section 4.3 of the Frequency Adjustment paper to obtain the scalar potential field of unwrapped phases.
λ is the regularization term, β scales the cross-channel phase constraints and freq_weights is a N-tuple that scales the frequencies.
"""
function anisotropic_phaseunwrap_rgb(waves :: NTuple{C,AbstractArray{Vector{Wave{T, N}}, N}}, τ; λ = 1e-6, β = 1e-3) where C where T<:AbstractFloat where N
    # Assemble and solve the multichannel linear system
    print("Phase unwrapping (RGB)... ")
    Ds = Vector{Vector{SparseMatrixCSC{T}}}(undef, C)
    fs = Vector{Vector{Vector{T}}}(undef, C)
    # Assemble the system for each channel
    for i in 1:C
        Ds[i], fs[i] = assemble_linear_system(waves[i], τ[i])
    end
    # Combine linear system data into a single array
    channeldata = [(waves=waves[i], D=Ds[i], f=fs[i]) for i in 1:C]
    # Compute cross-channel constraints
    CC,ccb = cross_channel_constraints(channeldata);
    D = [spblockdiagonal(getindex.(Ds, i)) for i = 1:N]
    f = [cat(getindex.(fs, i)...; dims=1) for i = 1:N]
    
    # Given the system solve via leastsquares
    # Multiply each backward-difference matrix by its transpose and then sum all with the cross-channel constraints and the regularization term
    A = (sum((X) -> X'X, D) + β*CC'CC + λ*I)
    # Allocate right side memory
    b = zeros(size(A, 1))
    u_all_axes_channels = map(1:N) do axis
        # Do the same for the n-th right side of the system
        b .= D[axis]'*f[axis];
        println("Solving the linear system for axis $(axis) of $(N).")
        # Solve
        u = @time(A \ b)
        # Propagate the computed phase values to the FrequencyAdjuster of each channel
        cumsiz = 0
        return map(1:C) do channel
            n = size(channeldata[channel].D[1], 2)
            cumsiz_old = cumsiz
            cumsiz += n
            u[(1:n) .+ cumsiz_old]
        end
    end
    # Propagate the computed phase values to the FrequencyAdjuster of each axis
    u_all = ntuple(Val(C)) do channel
        return map(1:N) do axis
            return u_all_axes_channels[axis][channel]
        end
    end
    GC.gc(true)
    println("done.")
    return u_all
end
