using .CUDA

export cu_adjust, cu_adjust_rgb
export cuFrequencyAdjuster
export cuFrequencyAdjusterMultichannel

"""
    cuFrequencyAdjuster{T <: AbstractFloat, N}

CUDA version of FrequencyAdjuster storing data on the device.
Can only perform exclusively isotropic or anisotropic reconstruction.
"""
struct cuFrequencyAdjuster{T <: AbstractFloat, N}
    h_frequency_adjuster :: FrequencyAdjuster{T, N}
    d_waves :: CuArray#{NTuple{M, T}, N, CUDA.Mem.DeviceBuffer}
    d_windowtimesdual :: CuArray{T, 1, CUDA.Mem.DeviceBuffer}
    d_input :: CuArray#{Union{Gray{T}, RGB{T}}, N, CUDA.Mem.DeviceBuffer}
    d_q :: CuArray{T, N, CUDA.Mem.DeviceBuffer}
    d_r :: CuArray{T, N, CUDA.Mem.DeviceBuffer}
end

"""
    cuFrequencyAdjusterMultichannel{T <: AbstractFloat, N}

CUDA version of FrequencyAdjusterMultichannel storing data on the device.
Can only perform exclusively isotropic or anisotropic reconstruction.
"""
const cuFrequencyAdjusterMultichannel{T, N, C} = NTuple{C, cuFrequencyAdjuster{T, N}}

"""
    cuFrequencyAdjuster(frequency_adjuster :: FrequencyAdjuster{T, N}; aniso=false)

Constructor of cuFrequencyAdjuster that stores host data from frequency_adjuster into the device.
If aniso is true then call anisotropic_phaseunwrap! to compute the phases, else call phaseunwrap!.
"""
function cuFrequencyAdjuster(frequency_adjuster :: FrequencyAdjuster{T, N}; aniso=false) where T <: AbstractFloat where N
    waves = frequency_adjuster.data[:waves]
    if aniso
        phases = anisotropic_phaseunwrap!(frequency_adjuster)
    else
        phases = [phaseunwrap!(frequency_adjuster)]
    end
    h_waves = fill(ntuple(x->T(0), length(phases)+1+N), maximum(length.(waves.parent)), size(waves)...)
    n = 1
    for I in CartesianIndices(size(waves))
        for (m, wave) in enumerate(waves.parent[I])
            h_waves[m, I.I...] = (get.(phases, n, T(0))..., wave.amplitude, wave.freqs...)
            n += 1
        end
    end
    d_waves = CuArray{NTuple{length(phases)+1+N, T}}(h_waves)
    d_input = CuArray(frequency_adjuster.data[:input])
    d_q = CuArray{T}(frequency_adjuster.data[:q])
    d_r = CUDA.zeros(T, size(frequency_adjuster.data[:input]))
    d_windowtimesdual = CuArray{T}(frequency_adjuster. window .* frequency_adjuster.dualwindow)
    cuFrequencyAdjuster{T, N}(frequency_adjuster, d_waves, d_windowtimesdual, d_input, d_q, d_r) :: cuFrequencyAdjuster{T, N}
end

"""
    cuFrequencyAdjusterMultichannel(frequency_adjuster :: FrequencyAdjusterMultichannel{T, N}; aniso=false)

Constructor of cuFrequencyAdjusterMultichannel that stores host data from frequency_adjuster into the device.
If aniso is true then call anisotropic_phaseunwrap! to compute the phases, else call phaseunwrap!.
"""
function cuFrequencyAdjusterMultichannel(
  frequency_adjuster :: FrequencyAdjusterMultichannel{T, N, C}
; aniso=false
) where T <: AbstractFloat where N where C
    if aniso
        anisotropic_phaseunwrap_rgb!(frequency_adjuster)
    else
        phaseunwrap_rgb!(frequency_adjuster)
    end
    return ntuple(3) do i
        return cuFrequencyAdjuster(frequency_adjuster[i])
    end
end

@inline function eval_wave(window::T, α::T, shift_x::T, shift_y::T, phase::T, amplitude::T, freq_v::T, freq_h::T)::T where T <: AbstractFloat
    return window * amplitude * cos(α * (phase + 2T(π) * (freq_h * shift_x + freq_v * shift_y)))
end

@inline function eval_wave_anisotropic(window::T, α::NTuple{2, T}, shift_x::T, shift_y::T, phase_v::T, phase_h::T, amplitude::T, freq_v::T, freq_h::T)::T where T <: AbstractFloat
    return window * amplitude * cos(α[1] * (phase_v + 2T(π) * freq_v * shift_y) + α[2] * (phase_h + 2T(π) * freq_h * shift_x))
end

@inline function eval_wave(window::T, α::T, shift::NTuple{N, T}, wave::NTuple{M, T})::T where T <: AbstractFloat where N where M
    return window * wave[2] * cos(α * (wave[1] + 2T(π) * sum(wave[3:end] .* shift)))
end

@inline function eval_wave_anisotropic(window::T, α::NTuple{N, T}, shift::NTuple{N, T}, wave::NTuple{M, T})::T where T <: AbstractFloat where N where M
    phase = α[1] * (wave[1] + 2T(π) * (wave[N+2] * shift[1]))
    for i = 2:N
        phase += α[i] * (wave[i] + 2T(π) * (wave[N+1+i] * shift[i]))
    end
    return window * wave[N+1] * cos(phase)
end

"""
    gpu_gather_reconstruct_adjusted_waves_kernel

Reconstruction kernel to recover the high-frequency content r detected in waves after adjusting the frequencies by α.

# Arguments
- `r::CuDeviceArray{T, 2}`: array to store the high-frequency content to be reconstructed
- `α::Union{T, NTuple{2, T}}`: adjustment factor
- `waves::CuDeviceArray`: detected waves
- `τ::Int32`: spacing between samples
- `windowtimesdual::CuDeviceVector{T}`: reconstruction window
- `window_size::Int32`: size of reconstruction window
- `half_num_waves_affecting_pixel::Int32`: half the size of the window in the τ-spaced space.
- `p0_y::Int32`: y offset from the first index
- `p0_x::Int32`: x offset from the first index
- `f::Function`: function to evaluate the color value of a wave
"""
function gpu_gather_reconstruct_adjusted_waves_kernel(
    r::CuDeviceArray{T, 2}, 
    α::Union{T, NTuple{2, T}}, 
    waves::CuDeviceArray,#{NTuple{4, T}, 3}, 
    τ::Int32, 
    windowtimesdual::CuDeviceVector{T}, 
    window_size::Int32, 
    half_num_waves_affecting_pixel::Int32,
    p0::NTuple{2, Int32},
    f::Function
) where T <: AbstractFloat
    # Compute id of this thread
    thread_id::Int32 = threadIdx().x + (blockIdx().x-1) * blockDim().x - 1
    if length(r) > thread_id
        # Get size of reconstructed image
        height::Int32, width::Int32 = size(r)
        # Compute (x, y) cartesian coordinate of pixel being computed by this thread assuming column major order
        y0::Int32 = thread_id % height + 1
        x0::Int32 = thread_id ÷ height + 1
        height -= 1
        width -= 1
        texcoord_y::T = (y0 - 1) / height
        texcoord_x::T = (x0 - 1) / width
        
        # nearestwaveind in the reduced image coordinate system.
        nearestwaveind_x::Int32 = (x0 - p0[2]) ÷ τ
        nearestwaveind_y::Int32 = (y0 - p0[1]) ÷ τ

        _shiftx::T = texcoord_x * width - x0 + T(0.5)
        _shifty::T = texcoord_y * height - y0 + T(0.5)
        
        for j::Int32 = -half_num_waves_affecting_pixel:half_num_waves_affecting_pixel
            for i::Int32 = -half_num_waves_affecting_pixel:half_num_waves_affecting_pixel
                # neighborind = (x, y)
                x::Int32 = nearestwaveind_x + j
                y::Int32 = nearestwaveind_y + i
                
                shift_x::T = x0 - (x * τ + p0[2])
                shift_y::T = y0 - (y * τ + p0[1])
                window_ind_x::Int32 = shift_x+(window_size ÷ 2)+1
                window_ind_y::Int32 = shift_y+(window_size ÷ 2)+1
                
                if( window_ind_x < 1 || window_ind_x > window_size ||
                    window_ind_y < 1 || window_ind_y > window_size)
                    continue
                end
                window::T = windowtimesdual[window_ind_x] * windowtimesdual[window_ind_y]
                
                # Convert to 1-indexing
                x += 1
                y += 1
                
                shift_x += _shiftx
                shift_y += _shifty
                
                for k::Int32 = 1:size(waves, 1)
                    r[y0, x0] += f(window, α, shift_x, shift_y, waves[ k, y, x ]...)
                end
            end
        end
    end
end

"""
    gpu_gather_reconstruct_adjusted_waves_kernel

Reconstruction kernel to recover the high-frequency content r detected in waves after adjusting the frequencies by α.

# Arguments
- `r::CuDeviceArray{T, N}`: array to store the high-frequency content to be reconstructed
- `α::Union{T, NTuple{N, T}}`: adjustment factor
- `waves::CuDeviceArray`: detected waves
- `τ::Int32`: spacing between samples
- `windowtimesdual::CuDeviceVector{T}`: reconstruction window
- `window_size::Int32`: size of reconstruction window
- `half_num_waves_affecting_pixel::Int32`: half the size of the window in the τ-spaced space.
- `p0::NTuple{N, Int32}`: y offset from the first index
- `f::Function`: function to evaluate the color value of a wave
"""
function gpu_gather_reconstruct_adjusted_waves_kernel(
    r::CuDeviceArray{T, N}, 
    α::Union{T, NTuple{N, T}}, 
    waves::CuDeviceArray,
    τ::Int32, 
    windowtimesdual::CuDeviceVector{T}, 
    window_size::Int32, 
    half_num_waves_affecting_pixel::Int32,
    p0::NTuple{N, Int32}, 
    window_indexing::CartesianIndices{N},
    f::Function
) where T <: AbstractFloat where N
    # Compute id of this thread
    thread_id::Int32 = threadIdx().x + (blockIdx().x-1) * blockDim().x
    # Get size of reconstructed image
    sz::NTuple{N, Int32} = size(r)
    if length(r) >= thread_id
        # Compute (x, y) cartesian coordinate of pixel being computed by this thread assuming column major order
        P::NTuple{N, Int32} = Tuple( CartesianIndices(sz)[thread_id] )
        sz = sz .- 1
        texcoord::NTuple{N, T} = @. (P - 1) / sz
        
        # nearestwaveind in the reduced image coordinate system.
        nearestwaveind::NTuple{N, T} = @. (P - p0) ÷ τ

        _shift::NTuple{N, T} = @. texcoord * sz - P + T(0.5)
        
        for I in window_indexing
            # neighborind = (x, y)
            X::NTuple{N, Int32} = nearestwaveind .+ Tuple(I)
            
            shift::NTuple{N, T} = @. P - (X * τ + p0)
            window_ind::NTuple{N, Int32} = @. shift+((window_size ÷ 2) + 1)
            
            if any(ind -> ind < 1 || ind > window_size, window_ind)
                continue
            end
            window::T = prod(ind -> windowtimesdual[ind], window_ind)
            
            # Convert to 1-indexing
            X = X .+ 1
            
            shift = shift .+ _shift
            
            for k::Int32 = 1:size(waves, 1)
                r[P...] += f(window, α, shift, waves[k, X...])
            end
        end
    end
end


"""
    cu_launch_kernel(len, func, args...)

Small utility function to launch a 1-dimensional kernel with at len threads for func(args...).
The optimal number of threads per block is computed by the occupancy API.
"""
function cu_launch_kernel(len, func, args...)
    kernel = @cuda launch=false func(args...)
    config = CUDA.launch_configuration(kernel.fun)
    threads = min(len, config.threads)
    blocks = cld(len, threads)
    kernel(args...; threads, blocks)
end

function cu_gather_reconstruct_adjusted_waves(F::cuFrequencyAdjuster{T, 2}
, α :: T
; func=eval_wave) where T <: AbstractFloat where N
    frequency_adjuster = F.h_frequency_adjuster
    
    num_waves_affecting_pixel::Int32 = frequency_adjuster.L ÷ frequency_adjuster.τ
    num_waves_affecting_pixel += (num_waves_affecting_pixel + 1) & 1
    half_num_waves_affecting_pixel::Int32 = num_waves_affecting_pixel ÷ 2
    
    cu_launch_kernel(length(F.d_r), gpu_gather_reconstruct_adjusted_waves_kernel, F.d_r, α, F.d_waves, Int32(frequency_adjuster.τ), F.d_windowtimesdual, Int32(frequency_adjuster.L), half_num_waves_affecting_pixel, Int32.(frequency_adjuster.data[:waves].offsets.+1), func)
    
    return F.d_r
end

function cu_gather_reconstruct_adjusted_waves(F::cuFrequencyAdjuster{T, N}
, α :: T
; func=eval_wave) where T <: AbstractFloat where N
    frequency_adjuster = F.h_frequency_adjuster
    
    num_waves_affecting_pixel::Int32 = frequency_adjuster.L ÷ frequency_adjuster.τ
    num_waves_affecting_pixel += (num_waves_affecting_pixel + 1) & 1
    half_num_waves_affecting_pixel::Int32 = num_waves_affecting_pixel ÷ 2
    
    cu_launch_kernel(length(F.d_r), gpu_gather_reconstruct_adjusted_waves_kernel, F.d_r, α, F.d_waves, Int32(frequency_adjuster.τ), F.d_windowtimesdual, Int32(frequency_adjuster.L), half_num_waves_affecting_pixel, Int32.(frequency_adjuster.data[:waves].offsets.+1), CartesianIndices(centered(zeros(ntuple(x -> 2half_num_waves_affecting_pixel+1, Val(N))))), func)
    
    return F.d_r
end

function cu_gather_reconstruct_adjusted_waves(F::cuFrequencyAdjuster{T, 2}
, α :: NTuple{N, T}
; func=eval_wave_anisotropic) where T <: AbstractFloat where N
    frequency_adjuster = F.h_frequency_adjuster
    
    num_waves_affecting_pixel::Int32 = frequency_adjuster.L ÷ frequency_adjuster.τ
    num_waves_affecting_pixel += (num_waves_affecting_pixel + 1) & 1
    half_num_waves_affecting_pixel::Int32 = num_waves_affecting_pixel ÷ 2
    
    cu_launch_kernel(length(F.d_r), gpu_gather_reconstruct_adjusted_waves_kernel, F.d_r, α, F.d_waves, Int32(frequency_adjuster.τ), F.d_windowtimesdual, Int32(frequency_adjuster.L), half_num_waves_affecting_pixel, Int32.(frequency_adjuster.data[:waves].offsets.+1), eval_wave_anisotropic)
    
    return F.d_r
end

function cu_gather_reconstruct_adjusted_waves(F::cuFrequencyAdjuster{T, N}
, α :: NTuple{N, T}
; func=eval_wave_anisotropic) where T <: AbstractFloat where N
    frequency_adjuster = F.h_frequency_adjuster
    
    num_waves_affecting_pixel::Int32 = frequency_adjuster.L ÷ frequency_adjuster.τ
    num_waves_affecting_pixel += (num_waves_affecting_pixel + 1) & 1
    half_num_waves_affecting_pixel::Int32 = num_waves_affecting_pixel ÷ 2
    
    cu_launch_kernel(length(F.d_r), gpu_gather_reconstruct_adjusted_waves_kernel, F.d_r, α, F.d_waves, Int32(frequency_adjuster.τ), F.d_windowtimesdual, Int32(frequency_adjuster.L), half_num_waves_affecting_pixel, Int32.(frequency_adjuster.data[:waves].offsets.+1), CartesianIndices(centered(zeros(ntuple(x -> 2half_num_waves_affecting_pixel+1, Val(N))))), eval_wave_anisotropic)
    
    return F.d_r
end

"""
    cu_adjust(s::Array{T, N}, α :: Union{Number, NTuple{N, Number}}; params...)

CUDA implementation of adjust.
"""
function cu_adjust(s::Array{T, N}
, α :: Union{Number, NTuple{N, Number}}
; params...) where T <: AbstractFloat where N
    frequency_adjuster = FrequencyAdjuster(s; params...)
    return cu_adjust(frequency_adjuster, α)
end

"""
    cu_adjust(s::Array{U, N}, α :: Union{Number, NTuple{N, Number}}; params...)

CUDA implementation of adjust.
"""
function cu_adjust(s::Array{U, N}
, α :: Union{Number, NTuple{N, Number}}
; params...) where U <: Color3{<:AbstractFloat} where N
    frequency_adjuster = FrequencyAdjuster(s; params...)
    return cu_adjust(frequency_adjuster, α)
end

"""
    cu_adjust(s::Array{U, N}, α :: Union{Number, NTuple{N, Number}}; params...)

CUDA implementation of adjust.
"""
function cu_adjust(s::Array{U, N}
, α :: Union{Number, NTuple{N, Number}}
; params...) where U <: Color1{<:AbstractFloat} where N
    frequency_adjuster = FrequencyAdjuster(s; params...)
    return cu_adjust(frequency_adjuster, α)
end

"""
    cu_adjust(frequency_adjuster::FrequencyAdjuster{T, N}, α :: Union{Number, NTuple{N, Number}}; params...)

CUDA implementation of adjust.
"""
function cu_adjust(frequency_adjuster::FrequencyAdjuster{T, N}
, α :: Union{Number, NTuple{N, Number}}
; params...) where T <: AbstractFloat where N
    if isa(α, Number)
        phaseunwrap!(frequency_adjuster)
    elseif isa(α, Tuple)
        anisotropic_phaseunwrap!(frequency_adjuster)
    end
    
    F = cuFrequencyAdjuster(frequency_adjuster)
    d_s_adjusted = cu_adjust(F, α; params...)
    
    h_s_adjusted = Array(d_s_adjusted)
    return h_s_adjusted
end

"""
    cu_adjust(F::cuFrequencyAdjuster{T, N}, α :: Union{Number, NTuple{N, Number}}; params...)

CUDA implementation of adjust. Returns a cuArray.
"""
function cu_adjust(F::cuFrequencyAdjuster{T, N}
, α :: Union{Number, NTuple{N, Number}}
; params...) where T <: AbstractFloat where N
    F.d_r .= 0
    d_r = cu_gather_reconstruct_adjusted_waves(F, T.(α); params...)
    d_q = F.d_q
    d_s0 = d_q .+ d_r
    if haskey(F.h_frequency_adjuster.data, :PCAstate)
        d_s_adjusted = FrequencyAdjustment.replace_normalized_principle_component(F.h_frequency_adjuster.data[:PCAstate], F.d_input, d_s0)
    else
        d_s_adjusted = d_q+d_r
    end
    return d_s_adjusted
end

"""
    cu_adjust_rgb(s::Array{T, N}, α::Union{Number, NTuple{N, Number}}; params...)

CUDA implementation of adjust_rgb.
"""
function cu_adjust_rgb(s::Array{U, N}
, α :: Union{Number, NTuple{N, Number}}
; params...) where U <: Color3{<:AbstractFloat} where N
    frequency_adjuster = FrequencyAdjusterMultichannel(s; params...)
    return cu_adjust_rgb(frequency_adjuster, α)
end

"""
    cu_adjust_rgb(frequency_adjuster::FrequencyAdjusterMultichannel{T, N, C}, α::Union{Number, NTuple{N, Number}}; params...)

CUDA implementation of adjust_rgb.
"""
function cu_adjust_rgb(frequency_adjuster::FrequencyAdjusterMultichannel{T, N, C}
, α :: Union{Number, NTuple{N, Number}}
; params...) where T <: AbstractFloat where N where C
    if isa(α, Number)
        phaseunwrap_rgb!(frequency_adjuster)
    elseif isa(α, Tuple)
        anisotropic_phaseunwrap_rgb!(frequency_adjuster)
    end
    
    d_s_adjusted = colorview(RGB, [cu_adjust(frequency_adjuster[i], α; params...) for i in 1:C]...)
    h_s_adjusted = Array(d_s_adjusted)
    
    return h_s_adjusted
end

"""
    cu_adjust_rgb(F::cuFrequencyAdjusterMultichannel{T, N, C}, α::Union{Number, NTuple{N, Number}}; params...)

CUDA implementation of adjust_rgb. Returns a cuArray.
"""
function cu_adjust_rgb(F::cuFrequencyAdjusterMultichannel{T, N, C}
, α::Union{Number, NTuple{N, Number}}
; params...) where C where T <: AbstractFloat where N
    colorview(RGB, [cu_adjust(F[i], α; params...) for i in 1:C]...)
end

adjust(F::cuFrequencyAdjuster{T, N}, α :: Union{Number, NTuple{N, Number}}; params...) where T <: AbstractFloat where N = cu_adjust(F, α; params...)
adjust_rgb(F::cuFrequencyAdjusterMultichannel{T, N, 3}, α :: Union{Number, NTuple{N, Number}}; params...) where T <: AbstractFloat where N = cu_adjust_rgb(F, α; params...)