module FrequencyAdjustment

using DSP
using FFTW
using Images
using Statistics
using AbstractFFTs
using SparseArrays
using LinearAlgebra
using ProgressLogging
using Requires

include("SpacedArrays.jl")
include("NormalizedColorPCA.jl")

using .SpacedArrays
using .NormalizedColorPCA

export adjust, adjust_rgb, adjust_lab
export FrequencyAdjuster, FrequencyAdjusterMultichannel
export phaseunwrap!, phaseunwrap_rgb!, anisotropic_phaseunwrap!, anisotropic_phaseunwrap_rgb!

# =================================
# Types
# =================================

"""
    Wave(a, b, phase, amplitude, a_remapped, b_remapped)

A wave of the form ``amplitude * cos( 2π * (a*x + b*y + phase) )``.  The values
of `{a,b}_remapped` represent the frequency-remapped horizontal and vertical
frequencies of the wave.
"""
struct Wave{T <: AbstractFloat, N}
    freqs :: NTuple{N, T}
    phase :: T
    amplitude :: T
end

"""
    FrequencyAdjuster{T <: AbstractFloat, N}

An object for adjusting frequencies of a N-D image of type T.
"""
mutable struct FrequencyAdjuster{T <: AbstractFloat, N}

    # Data computed during various phases of the algorithm
    data :: Dict{Symbol,Any}

    # Gaussian window std. dev.
    σ :: T

    # Reduction factor
    R :: T

    # Step size for spatial translations of the window
    τ :: Int
    
    # Max number of detected waves in a point
    max_waves :: Int

    # Radius of the spectral circle ℭ, equal to 0.4/R by default.
    spectralradius :: T

    # Frequency-domain std. dev. of the Gaussian window
    Σ :: T

    # Window for the Gabor transform
    window :: Vector{T}
    windowNd :: Array{T, N}
    dualwindow :: Vector{T}
    dualwindowNd :: Array{T, N}

    # DFT constants
    L :: Int
    rfftM :: Int

    # Precomputed indices
    indices_outside_spectralcircle :: Vector{Int}
    realDFT_to_DFT_indices :: Array{Int, N}
end

"""
    FrequencyAdjusterMultichannel{T <: AbstractFloat, N}

An object for adjusting frequencies of a N-D image of type T using the multichannel rgb optimization described in Section 4.3 of the Frequency Adjustment paper.
"""
const FrequencyAdjusterMultichannel{T, N, C} = NTuple{C,FrequencyAdjuster{T, N}}

Base.show(io::IO, S::FrequencyAdjuster{T}) where T = print(io, "FrequencyAdjuster{$T}")

include("detect_waves.jl")
include("phase_unwrapping.jl")

# =================================
# Constructors
# =================================

"""
    s̊ = compute(s::AbstractMatrix{T}; R::Number = 2, σ::Number = R/2) where T <: AbstractFloat

Perform spectral analysis of the monochromatic image `s`, considering that the
image will subsequently suffer a downscaling by a factor of `R` in both
dimensions.  Frequency detection is performed with a Gabor space-frequency
decomposition whose Gaussian window has a spatial standard deviation of `σ`.
"""
function FrequencyAdjuster(
  s :: AbstractArray{T, N}
; R :: Number = 2
, σ :: Number = 0.5*R
, τ :: Integer = ceil(Int,σ)
, max_waves :: Integer = 10
) where T <: AbstractFloat where N

    # Type conversion
    R = T(R)
    σ = T(σ)
    τ = Int(τ)
    max_waves = Int(max_waves)

    # Detection parameters
    spectralradius = T(0.4) / R
    window = map(T, gaussian_window(σ)) :: Vector{T}
    dualwindow = compute_dual_window(window, τ)
    L = length(window)
    Σ = T(inv(2π * σ))

    # Compute a N-dimensional gaussian window of size L
    windowNd = copy(window)
    dualwindowNd = copy(dualwindow)
    for i=2:N
        v = ones(Int, N)
        v[i] = L
        # Tensor product of a i-D Gaussian window with a 1-D Gaussian window to compute a i+1-D Gaussian window
        windowNd = windowNd.*reshape(window, v...)
        dualwindowNd = dualwindowNd.*reshape(dualwindow, v...)
    end

    # Number of samples in the DFT (fft) and real DFT (rfft)
    rfftM = L÷2 + 1

    # DFT sample frequencies
    U = cat([map(T, FFTW.rfftfreq(L))], [map(T, FFTW.fftfreq(L)) for i=2:N],dims=1)

    # Frequencies of a digital signal uniformely sampled in a LxL...xL region
    freqs = [map((u, n) -> u[n], U[end:-1:1], i[end:-1:1]) for i in Iterators.product(map((x) -> 1:x, length.(U))...)]
    
    # Indices of waves with frequencies outside the spectral circle
    indices_outside_spectralcircle = [ i for i in 1:lastindex(freqs) if norm(freqs[i]) .>=  spectralradius ]
    realDFT_to_DFT_indices = generate_DFT_to_realDFT_indices(L, Val(N))

    # Object constructor
    S = FrequencyAdjuster(
        Dict{Symbol,Any}(),
        σ, R, τ, max_waves, spectralradius, Σ
        , window , windowNd , dualwindow , dualwindowNd
        , L , rfftM , indices_outside_spectralcircle
        , realDFT_to_DFT_indices
    ) :: FrequencyAdjuster{T, N}

    # Detect local waves in the signal (Section 4.1 and Appendix B of the Spectral Remapping paper)
    detect_waves!(S, s)

    # Save input image s
    S.data[:input] = s
    return S
end

const Color1 = Color{W,1} where W

"""
    s̊ = FrequencyAdjuster([T,] s::AbstractMatrix{U}; R::Number = 2, σ::Number = R/2) where U <: Color{W,1}

Perform spectral analysis of the one-channel image `s`. Computations are
performed internally with floating point type T (default Float64).
"""
function FrequencyAdjuster(
  ::Type{T}
, s::AbstractArray{U, N}
; params...
) where T <: AbstractFloat where U <: Color1 where N
    S = FrequencyAdjuster(T.(s); params...)
    S.data[:input] = s
    return S
end

FrequencyAdjuster(s::Array{U, N}; params...) where U <: Color1 where N = FrequencyAdjuster(Float64, s; params...)

"""
    s̊ = compute([T,] s::Matrix{U}; R::Number = 2, σ::Number = R/2) where U <: Color{W,3}

Perform spectral analysis of the three-channel color image `s` using principal
component analysis. Computations are performed internally with floating
point type T (default Float64).
"""
function FrequencyAdjuster(
    :: Type{T}
, s :: Array{U, N}
; params...
) where T <: AbstractFloat where U <: Color3 where N
    PCAstate, η = normalized_principle_component(T, s)
    S = FrequencyAdjuster(η; params...)
    S.data[:PCAstate] = PCAstate
    S.data[:input] = s
    return S
end

FrequencyAdjuster(s::Array{U, N}; params...) where U <: Color3 where N = FrequencyAdjuster(Float64, s; params...)

function FrequencyAdjusterMultichannel(
  s :: Array{<:Color3{T}, N}
; detection_parameters...
) where T <: AbstractFloat where N
    return ntuple(3) do i
        comp = (comp1, comp2, comp3)[i]
        s_channel = comp.(s)
        return FrequencyAdjuster(s_channel; detection_parameters...)
    end
end

Wave{T, N}(wave::Wave{F, N}) where T <: AbstractFloat where F <: AbstractFloat where N = Wave(T.(wave.freqs), T(wave.phase), T(wave.amplitude))

function FrequencyAdjuster{T, N}(S::FrequencyAdjuster{F, N}) where T  where F  where N
    data = Dict{Symbol, Any}()
    if haskey(S.data, :waves)
        data[:waves] = map(waves -> Wave{T, N}.(waves), S.data[:waves])
    end
    if haskey(S.data, :input)
        data[:input] = base_color_type(eltype(S.data[:input])){T}.(S.data[:input])
    end
    if haskey(S.data, :q)
        data[:q] = T.(S.data[:q])
    end
    if haskey(S.data, :u_all)
        data[:u_all] = T.(S.data[:u_all])
    end
    if haskey(S.data, :u_all_aniso)
        data[:u_all_aniso] = map(phases -> T.(phases), S.data[:u_all_aniso])
    end
    if haskey(S.data, :PCAstate)
        data[:PCAstate] = NormalizedColorPCAState{T}(S.data[:PCAstate])
    end
    FrequencyAdjuster{T, N}(
        data,
        T(S.σ), T(S.R), S.τ, S.max_waves, T(S.spectralradius), T(S.Σ)
        , T.(S.window), T.(S.windowNd), T.(S.dualwindow), T.(S.dualwindowNd)
        , S.L , S.rfftM , copy(S.indices_outside_spectralcircle)
        , copy(S.realDFT_to_DFT_indices)
    ) :: FrequencyAdjuster{T, N}
end

# =================================
# Methods
# =================================

"""
    s̊ = _scatter_reconstruct_adjusted_waves(S::FrequencyAdjuster{T, N}, adjustmentfn :: Function)

Reconstruct the image scaling the wave w::Wave{T, N} phase (and frequency)
at position I::CartesianIndex{N} and wave index wave_idx::Int at I by
adjustmentfn(w, I, wave_idx)::T for each wave detected in S.
"""
function _scatter_reconstruct_adjusted_waves(
  S :: FrequencyAdjuster{T, N}
, adjustmentfn :: Function
) where T <: AbstractFloat where Func <: Function where N

    # Local variables
    # N-D array of detected waves with up to S.max_waves waves per sample
    waves = S.data[:waves].parent::Array{Vector{Wave{T, N}}, N}
    # 1-D array of phases indexed by waves_linear_counting function defined below
    phases = S.data[:u_all]::Vector{T}
    # Reconstruction window
    windowtimesdual::Array{T, N} = S.windowNd .* S.dualwindowNd ::Array{T, N}

    # Local window coordinates. Its center is at (0,0)
    l  = S.L ÷ 2

    # Allocate memory for the output vector `r`
    r = zeros(T, padsize(size(S.data[:q]), l))

    # Compute indexing function
    waves_linear_counting = map(length, waves)[:]
    waves_linear_counting = cumsum(waves_linear_counting) .- waves_linear_counting .+ 1
    waves_linear_counting = reshape(waves_linear_counting, size(waves))
    wave_indexing_function = (I, freq_idx, waves_linear_counting) -> begin
        waves_linear_counting[I] + freq_idx - 1
    end

    # Mutex used for updating reconstructed image
    mutex = Threads.SpinLock()
    # Count of curent processed waves
    count = Threads.Atomic{Int}(0)
    # Total number of waves
    total = prod(size(waves))
    # Local reconstructed waves to be summed to global reconstruction
    localrs = Array{T, N}[zeros(T, size(windowtimesdual)) for i in 1:Threads.nthreads()]
    # List of N-D arrays with phase of the current wave at its windowed neighborhood for each thread
    window_phase_list = Array{T, N}[zeros(T, size(windowtimesdual)) for i in 1:Threads.nthreads()]
    # Shifts from the center of the window
    window_shift = map((x) -> x.I, collect(CartesianIndices(centered(windowtimesdual))))
    # Start progress logging block
    @withprogress name="Reconstructing adjusted waves..." begin
        # For each window
        Threads.@threads for I in CartesianIndices(size(waves))
            @inbounds begin
                # Position of the current window,
                # considering the origin (1,1) on the top-left of the un-padded image.
                P = CartesianIndex(S.τ.*(I.I .- 1))

                # Local thread variables
                # Get local reconstruction window and set values to 0
                localr = localrs[Threads.threadid()]
                localr .= T(0)
                # Window with phase of wave at each coordinate of the detection window
                window_phase = window_phase_list[Threads.threadid()]

                # For each wave inside the window
                for (wave_idx, w) in enumerate(waves[I])
                    # Compute phase at center of window and adjustment factor α
                    phase::T = phases[ wave_indexing_function(I, wave_idx, waves_linear_counting) ]
                    α = adjustmentfn(w, I, wave_idx)
                    # Estimate adjusted phase at each point of window based on frequency and shift from center of window
                    # equivalent to part inside cosine of Equation 20 of the Frequency Adjustment paper
                    broadcast!((δ) -> α * (phase + T(2π)*dot(w.freqs, δ)), window_phase, window_shift)
                    # Reconstructs the wave given the adjusted phase, amplitude and appropriate reconstruction window
                    # remaining of Equation 20 of the Frequency Adjustment paper
                    broadcast!((r, rec_window, phase) -> r + rec_window * w.amplitude * cos(phase),
                                    localr, localr, windowtimesdual, window_phase)
                end
                # Add to global reconstruction
                lock(mutex) do
                    for J in CartesianIndices(size(localr))
                        r[P + J] += localr[J]
                    end
                end
                count[] += 1 # Atomic increment
                if Threads.threadid() == 1
                    # Update progress
                    @logprogress count[] / total
                end
            end
        end
    end

    # Remove boundary padding
    @inbounds return @view(r[map((s) -> 2l+1:s-2l, size(r))...]) :: SubArray{T,N}
end

"""
    s̊ = _gather_reconstruct_adjusted_waves(S::FrequencyAdjuster{T, N}, adjustmentfn :: Function)

Reconstruct the image scaling the wave w::Wave{T, N} phase (and frequency)
at position I::CartesianIndex{N} and wave index wave_idx::Int at I by
adjustmentfn(w, I, wave_idx)::T for each wave detected in S.
"""
function _gather_reconstruct_adjusted_waves(S::FrequencyAdjuster{T, N}
, adjustmentfn :: Function
; siz = size(S.data[:input])) where T <: AbstractFloat where N
    waves = S.data[:waves]
    phases = S.data[:u_all]::Vector{T}
    recwindow = centered(S. windowNd .* S.dualwindowNd)
    num_waves_affecting_pixel = S.L .÷ S.τ
    num_waves_affecting_pixel += (num_waves_affecting_pixel+1) % 2
    
    # Compute indexing function
    waves_linear_counting = map(length, waves.parent)[:]
    waves_linear_counting = cumsum(waves_linear_counting) .- waves_linear_counting .+ 1
    waves_linear_counting = reshape(waves_linear_counting, size(waves.parent))
    wave_indexing_function = (I, freq_idx) -> begin
        waves_linear_counting[I] + freq_idx - 1
    end
    
    inv_ratio = size(S.data[:input]) ./ siz
    r = zeros(T, siz)
    helper = centered(zeros(ntuple(n -> num_waves_affecting_pixel, Val(N))))
    neighborhood = CartesianIndices(helper)
    # For each window
    count = Threads.Atomic{Int}(0)
    total = prod(siz)
    @withprogress name = "Reconstructing" begin
        Threads.@threads for dst_pixelind in CartesianIndices(siz)
            src_pixelind_float = dst_pixelind.I .* inv_ratio
            src_pixelind = CartesianIndex(trunc.(Int, src_pixelind_float))
            nearestwaveind = SpacedArrays.nearestvalid(waves, src_pixelind)
            centershift = src_pixelind - nearestwaveind
            for δ in neighborhood
                neighborind = SpacedArrays.spacedstep(waves, nearestwaveind, δ.I...)
                recwindowind = δ * S.τ - centershift
                if checkbounds(Bool, waves, neighborind) && checkbounds(Bool, recwindow, recwindowind)
                    shift = src_pixelind_float .- neighborind.I
                    I = CartesianIndex((neighborind.I.-waves.offsets).÷S.τ.+1)
                    for (waveind, w) in enumerate(waves[neighborind])
                        phase::T = phases[ wave_indexing_function(I, waveind) ]
                        α = adjustmentfn(w, I, waveind)
                        r[dst_pixelind] += recwindow[recwindowind] * w.amplitude *
                                cos(α * (phase + T(2π)*dot(w.freqs, shift))) :: T
                    end
                end
            end
            count[] += 1 # Atomic increment
            if Threads.threadid() == 1
                @logprogress count[] / total
            end
        end
    end
    return r
end

"""
    s̊ = anisotropic_scatter_reconstruct_adjusted_waves(S::FrequencyAdjuster{T, N}, adjustmentfn :: Function)

Reconstruct the image scaling the wave w::Wave{T, N} phase (and frequency)
in standard axes at position I::CartesianIndex{N} and wave index wave_idx::Int
at I by adjustmentfn(w, I, wave_idx)::Vector{T} for each wave detected in S.
"""
function anisotropic_scatter_reconstruct_adjusted_waves(
  S :: FrequencyAdjuster{T, N}
, adjustmentfn :: Function
) where T <: AbstractFloat where Func <: Function where N

    # Local variables
    # N-D array of detected waves with up to S.max_waves waves per sample
    waves = S.data[:waves].parent::Matrix{Vector{Wave{T, N}}}
    # 1-D array of phases indexed by waves_linear_counting function defined below
    phases = S.data[:u_all_aniso]::Vector{Vector{T}}
    # Reconstruction window
    windowtimesdual::Matrix{T} = S.windowNd .* S.dualwindowNd ::Matrix{T}

    # Local window coordinates. Its center is at (0,0)
    l  = S.L ÷ 2

    # Allocate memory for the output vector `r`
    r = zeros(T, padsize(size(S.data[:q]), l))

    # Compute indexing function
    waves_linear_counting = map(length, waves)[:]
    waves_linear_counting = cumsum(waves_linear_counting) .- waves_linear_counting .+ 1
    waves_linear_counting = reshape(waves_linear_counting, size(waves))
    wave_indexing_function = (I, freq_idx) -> begin
        waves_linear_counting[I] + freq_idx - 1
    end

    # Mutex used for updating reconstructed image
    mutex = Threads.SpinLock()
    # Count of curent processed waves
    count = Threads.Atomic{Int}(0)
    # Total number of waves
    total = prod(size(waves))
    # Local reconstructed wave to be summed to global reconstruction
    localrs = Array{T, N}[zeros(T, size(windowtimesdual)) for i in 1:Threads.nthreads()]
    # Shifts from the center of the window
    window_shift = map((x) -> x.I, collect(CartesianIndices(centered(windowtimesdual))))
    # List of N-D arrays with phase of the current wave at its windowed neighborhood for each thread
    window_phase_list = Array{T, N}[zeros(T, size(windowtimesdual)) for i in 1:Threads.nthreads()]
    # List of 1-D arrays with the phase for each axis at the center of the current wave for each thread
    local_phase = [zeros(T, N) for i in 1:Threads.nthreads()]
    local_corrected_phase = [zeros(T, N) for i in 1:Threads.nthreads()]
    # Start progress logging block
    @withprogress name="Reconstructing adjusted waves..." begin
        # For each window
        Threads.@threads for I in CartesianIndices(size(waves))
            @inbounds begin
                P = (S.τ*(I - CartesianIndex(1,1))).I

                # Local thread variables
                # Get local reconstruction window and set values to T(0)
                localr = localrs[Threads.threadid()]
                localr .= T(0)
                # Window with phase of wave at each coordinate of the detection window
                window_phase = window_phase_list[Threads.threadid()]
                # Pre-allocated phase vector
                phase = local_phase[Threads.threadid()]
                # Pre-allocated corrected-phase vector
                corrected_phase = local_corrected_phase[Threads.threadid()]

                # For each wave inside the window
                for (wave_idx, w) in enumerate(waves[I])
                    # Compute the phase index of the current wave and store into the vector phase
                    phase_ind = wave_indexing_function(I, wave_idx)
                    for n in 1:N
                        phase[n] = phases[n][phase_ind]
                    end
                    # Compute the adjustment functional α for wave w at cartesian index I and wave index wave_idx
                    α = adjustmentfn(w, I, wave_idx)
                    # For each coordinate in the window, get the shift from the center of the window and
                    for (i, δ) in enumerate(window_shift)
                        for n in 1:N
                            # compute the phase at that coordinate given the shift and wave frequency
                            corrected_phase[n] = phase[n] + T(2π) * (w.freqs[n] * δ[n])
                        end
                        # then adjust and store the phase after applying the α functional into the corrected phase
                        window_phase[i] = dot(α, corrected_phase)
                    end
                    # Reconstructs the wave given the adjusted phase, amplitude and appropriate reconstruction window
                    # remaining of Equation 20 of the Frequency Adjustment paper
                    map!((r, rec_window, phase) -> r + rec_window * w.amplitude * cos(phase),
                                    localr, localr, windowtimesdual, window_phase)
                end
                # Add to global reconstruction
                lock(mutex) do
                    r[map((n, l) -> P[n]+1:P[n]+l, 1:N, size(windowtimesdual))...] .+= localr
                end
                count[] += 1 # Atomic increment
                if Threads.threadid() == 1
                    # Update progress
                    @logprogress count[] / total
                end
            end
        end
    end

    # Remove boundary padding
    @inbounds return @view(r[map((s) -> 2l+1:s-2l, size(r))...]) :: SubArray{T,N}
end

"""
    phaseunwrap!(S::FrequencyAdjuster{T, N}, adjustmentfn :: Function)

Unwrap the phases detected in S by integrating the frequencies using the least-squares
approach described in Section 4 (Phase Unwrapping in Two Dimensions) of the Frequency
Adjustment paper.
"""
function phaseunwrap!(S::FrequencyAdjuster{T, N}; params...) where T <: AbstractFloat where N
    # Use previously computed varying phase. If it has not been computed, do it
    # now and store in S.data.
    u_all = get!(S.data, :u_all) do
        phaseunwrap(S.data[:waves], S.τ; params...)
    end
end

"""
    phaseunwrap_rgb!(S::FrequencyAdjuster{T, N}, adjustmentfn :: Function)

Unwrap the phases detected in S by integrating the frequencies using the least-squares
approach described in Section 4 (Phase Unwrapping in Two Dimensions) of the Frequency
Adjustment paper.
"""
function phaseunwrap_rgb!(S :: FrequencyAdjusterMultichannel{T, N, C}; params...) where C where T<:AbstractFloat where N
    # Use previously computed varying phase. If it has not been computed, do it
    # now and store in S.data.
    if all((S_C) -> !haskey(S_C.data, :u_all), S)
        u_all = phaseunwrap_rgb(ntuple((channel) -> S[channel].data[:waves], Val(C)),
                                ntuple((channel) -> S[channel].τ, Val(C));
                                params...)
        for (channel, u_all_channel) in enumerate(u_all)
            S[channel].data[:u_all] = u_all_channel
        end
    end
    return ntuple((channel) -> S[channel].data[:u_all], C)
end

"""
    anisotropic_phaseunwrap(S::FrequencyAdjuster{T, N}, adjustmentfn :: Function)

Unwrap the phases detected in S by integrating the frequencies independently for each
axes using the least-squares method described in the Thesis.
"""
function anisotropic_phaseunwrap!(S::FrequencyAdjuster{T, N}; params...) where T <: AbstractFloat where N
    # Use previously computed varying phase. If it has not been computed, do it
    # now and store in S.data.
    u_all_aniso = get!(S.data, :u_all_aniso) do
        anisotropic_phaseunwrap(S.data[:waves], S.τ; params...)
    end
    return u_all_aniso
end

"""
    anisotropic_phaseunwrap_rgb(S::FrequencyAdjuster{T, N}, adjustmentfn :: Function)

Unwrap the phases detected in S by integrating the frequencies independently for each
axes using the least-squares method described in the Thesis.
"""
function anisotropic_phaseunwrap_rgb!(S :: FrequencyAdjusterMultichannel{T, N, C}; params...) where C where T<:AbstractFloat where N
    # Use previously computed varying phase. If it has not been computed, do it
    # now and store in S.data.
    if all((S_C) -> !haskey(S_C.data, :u_all), S)
        u_all_aniso = anisotropic_phaseunwrap_rgb(ntuple((channel) -> S[channel].data[:waves], Val(C)),
                                                  ntuple((channel) -> S[channel].τ, Val(C));
                                                  params...)
        for channel in 1:C
            S[channel].data[:u_all_aniso] = u_all_aniso[channel]
        end
    end
    return ntuple((channel) -> S[channel].data[:u_all_aniso], C)
end

"""
    s̊ = reconstruct(S::FrequencyAdjuster{T, N}, adjustmentfn :: Function)

Unwrap the phases of S if not already unwrapped and then reconstructs
the image scaling the wave w::Wave{T, N} phase (and frequency)
at position I::CartesianIndex{N} and wave index wave_idx::Int at I by
adjustmentfn(w, I, wave_idx)::T for each wave detected in S.
"""
function reconstruct(S::FrequencyAdjuster{T, N}, adjustmentfn::Function) where T <: AbstractFloat where N
    u_all = phaseunwrap!(S)
    _scatter_reconstruct_adjusted_waves(S, adjustmentfn)
end

"""
    s̊ = adjust(S::FrequencyAdjuster{T, N}, adjustmentfn::Function)

Scales the frequencies of waves detected in S by adjustmentfn(w, I, wave_idx)
for every wave w at cartesian index I detected in wave_idx order.
"""
function adjust(S::FrequencyAdjuster{T, N}, adjustmentfn::Function) where T <: AbstractFloat where N
    # Store the input image s and the residual q from S
    s = S.data[:input]
    q = S.data[:q]
    # Store into r the high-frequency content from the detected waves in S after applying the adjustment function adjustmentfn
    r = reconstruct(S, adjustmentfn)
    # If S has a PCAstate data stored
    if haskey(S.data, :PCAstate)
        # use it to transform the residual q plus the reconstructed high-frequency content r
        # from the PCA defined space back to the RGB color space
        s0 = replace_normalized_principle_component(S.data[:PCAstate], s, q+r)
    else
        # otherwise the image is a single channel image that can be obtained by summing the residual and high-frequency content
        s0 = eltype(s).(q+r)
    end
    # Then return the adjusted image
    return s0
end

"""
    s̊ = adjust(S::FrequencyAdjuster{T, N}, α::Number)

Scales the frequencies of waves detected in S by alpha.
"""
function adjust(S::FrequencyAdjuster{T, N}, α::Number) where T <: AbstractFloat where N
    fα = convert(T, α)
    adjust(S, (a,b,c) -> return fα)
end

"""
    s̊ = adjust(S::FrequencyAdjuster{T, N}, α::NTuple{N, Number})

Scales the frequencies of waves detected in S at standard axis n by α[n].
"""
function adjust(S::FrequencyAdjuster{T, N}, α::NTuple{N, Number}) where T <: AbstractFloat where N
    fα = T.(α)
    s = S.data[:input]
    q = S.data[:q]
    anisotropic_phaseunwrap!(S)
    r = anisotropic_scatter_reconstruct_adjusted_waves(S, (a, b, c) -> return fα)
    if haskey(S.data, :PCAstate)
        s0 = replace_normalized_principle_component(S.data[:PCAstate], s, q+r)
    else
        s0 = eltype(s).(q+r)
    end
    return s0
end

"""
    s̊ = adjust(s :: Array{U, N}, α::Union{Number, NTuple{N, Number}}; params...)

Scales the frequencies of waves in s at standard axis n by α[n] if α::NTuple{N, number} or α if α::Number.
"""
function adjust(s::AbstractArray{T, N}, α::Union{Number, NTuple{N, Number}};params...) where T <: AbstractFloat where N
    S = FrequencyAdjuster(s; params...)
    adjust(S, α)
end

"""
    s̊ = adjust(s :: Array{U, N}, α::Union{Number, NTuple{N, Number}}; params...)

Scales the frequencies of waves in s at standard axis n by α[n] if α::NTuple{N, number} or α if α::Number.
"""
function adjust(s :: Array{U, N}, α::Union{Number, NTuple{N, Number}}; params...) where U <: Color3{<:AbstractFloat} where N
    S = FrequencyAdjuster(s; params...)
    adjust(S, α)
end

"""
    s̊ = adjust(S::FrequencyAdjuster{T, N}, α::NTuple{N, Number})

Scales the frequencies of waves detected in S at standard axis n by α[n].
"""
function adjust(s :: Array{U, N}, α::Union{Number, NTuple{N, Number}}; params...) where U <: Color1{<:AbstractFloat} where N
    S = FrequencyAdjuster(s; params...)
    adjust(S, α)
end

"""
    s̊ = adjust(S::NTuple{C,FrequencyAdjuster{T, N}})

Adjusts the frequencies of waves detected in S separately for each channel.
"""
function adjust_lab(S::NTuple{C,FrequencyAdjuster{T, N}}, α::Union{Number, NTuple{N, Number}}) where C where T <: AbstractFloat where N
    colorview(Lab, [FrequencyAdjustment.adjust(S[i], α) for i in 1:C]...)
end

"""
    s̊ = adjust_rgb(S::NTuple{C,FrequencyAdjuster{T, N}})

Adjusts the frequencies of waves detected in S separately for each channel.
"""
function adjust_rgb(S::FrequencyAdjusterMultichannel{T, N, C}, α::Union{Number, NTuple{N, Number}}) where C where T <: AbstractFloat where N
    phaseunwrap_rgb!(S)
    colorview(RGB, [FrequencyAdjustment.adjust(S[i], α) for i in 1:C]...)
end

"""
    s̊ = adjust_rgb(s :: Array{U, N}, α::Number; params...)

Adjusts the frequencies of waves in s by α separately for each channel.
"""
function adjust_rgb(s :: Array{U, N}, α::Number; params...) where U <: Color3 where N
    S = ntuple(3) do i
        fn = (comp1,comp2,comp3)[i]
        s_channel = fn.(s)
        S = FrequencyAdjuster(s_channel; params...);
    end;
    phaseunwrap_rgb!(S)
    adjust_rgb(S, α)
end

"""
    s̊ = adjust_rgb(s :: Array{U, N}, α::Number; params...)

Adjusts the frequencies of waves in s by α separately for each channel.
"""
function adjust_rgb(s :: Array{U, N}, α::NTuple{N, Number}; params...) where U <: Color3 where N
    S = ntuple(3) do i
        fn = (comp1,comp2,comp3)[i]
        s_channel = fn.(s)
        S = FrequencyAdjuster(s_channel; params...);
    end;
    anisotropic_phaseunwrap_rgb!(S)
    adjust_rgb(S, α)
end

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("CUDA_reconstruction.jl")
end

end
