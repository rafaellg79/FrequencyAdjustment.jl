export WaveIterator

struct WaveIterator{T}
    waves :: SpacedArray{Vector{W}} where W <: Wave{T}
    cli   :: CompactLinearIndices
end

WaveIterator(waves) = WaveIterator(waves, CompactLinearIndices(waves))

function Base.iterate(wi::WaveIterator)
    if wi.cli.lastindex == 1
        return nothing # empty
    else
        iter = Iterators.Stateful(SpacedCartesianIndices(wi.waves))
        local pixelind
        for pixelind in iter
            if length(wi.waves[pixelind]) > 0
                waveind = 1
                wavelinearind = wi.cli[pixelind,waveind]
                wave  = wi.waves[pixelind][waveind]
                data  = (wave, pixelind, waveind, wavelinearind)
                state = (iter, pixelind, waveind)
                return (data, state)
            end
        end
    end
end

function Base.iterate(wi::WaveIterator, state)
    #wave, pixelind, waveind, wavelinearind = state
    iter, pixelind, waveind = state
    if length(wi.waves[pixelind]) > waveind
        # Advance to next wave in the same pixel
        newpixelind = pixelind
        newwaveind = waveind + 1
        wave = wi.waves[newpixelind][newwaveind]
        newwavelinearind = wi.cli[newpixelind,newwaveind]
        data  = (wave, newpixelind, newwaveind, newwavelinearind)
        state = (iter, pixelind, waveind)
        return (data, state)
    else
        # Advance to the first wave of the next pixel that has a wave
        for newpixelind in iter
            if length(wi.waves[newpixelind]) > 0
                newwaveind = 1
                wave = wi.waves[newpixelind][newwaveind]
                newwavelinearind = wi.cli[newpixelind,newwaveind]
                data  = (wave, newpixelind, newwaveind, newwavelinearind)
                state = (iter, newpixelind, newwaveind)
                return (data, state)
            end
        end
        return nothing
    end
end

function Base.length(wi::WaveIterator)
    return wi.cli.lastindex - 1
end

eachwave(waves) = Iterators.flatten(waves)
