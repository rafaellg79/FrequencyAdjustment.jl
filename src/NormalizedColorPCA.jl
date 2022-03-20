module NormalizedColorPCA

using Images
using Statistics
using ColorTypes
using LinearAlgebra
using Requires

export normalized_principle_component
export replace_normalized_principle_component
export NormalizedColorPCAState

struct NormalizedColorPCAState{T}
    V :: Matrix{T} # Eigenvector orthogonal basis
    ma :: T # Normalization value (maximum)
    mi :: T # Normalization value (minimum)
end

NormalizedColorPCAState{T}(p::NormalizedColorPCAState{F}) where T <: AbstractFloat where F <: AbstractFloat = NormalizedColorPCAState{T}(T.(p.V), T(p.ma), T(p.mi))

function Base.show(io::IO, p::NormalizedColorPCAState{T}) where T
    println(io, "NormalizedColorPCAState{$T} with:")
    println(io, "= Normalization constants:")
    println(io, "  min = $(p.mi)")
    println(io, "  max = $(p.ma)")
    println(io, "= Eigenvectors:")
    show(io, MIME("text/plain"), p.V)
end

function pca(
    :: Type{T}
, s :: Array{U}
) where T <: AbstractFloat where U <: Color3
    C = zeros(T,3,3) # RGB covariance matrix
    meancolor = mean(base_color_type(U){T}, s)
    @inbounds @simd for i in eachindex(s)
        diff = s[i] - meancolor
        r = comp1(diff)
        g = comp2(diff)
        b = comp3(diff)
        C[1,1] += r*r; C[2,2] += g*g; C[3,3] += b*b;
        C[1,2] += r*g; C[1,3] += r*b; C[2,3] += g*b;
    end
    C[2,1] = C[1,2]; C[3,1] = C[1,3]; C[3,2] = C[2,3];
    V = eigvecs(C)
end

function analysis(
  V :: Matrix{T}
, s :: Array{U}
) where T <: AbstractFloat where U <: Color3
    u = similar(s, T)
    @inbounds @simd for i in eachindex(s)
        # Project onto the third eigenvector (associated with the largest eigenvalue)
        u[i] = V[:,3]' * [comp1(s[i]),comp2(s[i]),comp3(s[i])]
    end
    u
end

function synthesis(
  V :: Matrix{Float16}
, s :: Array{U}
, s0 :: Array{Float16}
) where U <: Color3
    synthesis(Float32.(V), s, Float32.(s0))
end

function synthesis(
  V :: Matrix{T}
, s :: Array{U}
, s0 :: Array{T}
) where T <: AbstractFloat where U <: Color3
    u = T.(channelview(vec(s)))
    o = base_color_type(U){T}.(s)
    v = channelview(vec(o))
    BLAS.gemm!('C', 'N', one(T), V, v, zero(T), u)
    @inbounds u[3,:] = vec(s0)
    BLAS.gemm!('N', 'N', one(T), V, u, zero(T), v)
    eltype(s).(clamp01.(o))
end

function normalized_principle_component(::Type{T}, s::Array{U}) where T <: AbstractFloat where U <: Color3
    V = pca(T,s)
    η = analysis(V,s)
    ma,mi = extrema(η)
    p = NormalizedColorPCAState(V,ma,mi)
    map!( x->(x-mi)/(ma-mi), η, η )
    return (p, η)
end

normalized_principle_component(s) = normalized_principle_component(Float64, s)

function replace_normalized_principle_component(
  p :: NormalizedColorPCAState{T}
, s :: Array{U}
, η :: Array{T}
) where T <: AbstractFloat where U <: Color3
    map!( x->x*(p.ma-p.mi)+p.mi, η, η)
    s0 = synthesis(p.V,s,η)
end

using Requires
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" synthesis(
      V :: CUDA.CuArray{T}
    , s :: CUDA.CuArray{U}
    , s0 :: CUDA.CuArray{T}
    ) where T <: AbstractFloat where U <: Color3 = begin
        u = T.(channelview(vec(s)))
        #o = base_color_type(U){T}.(s)
        v = T.(channelview(vec(s)))
        CUDA.CUBLAS.gemm!('C', 'N', one(T), V, v, zero(T), u)
        @inbounds u[3,:] .= vec(s0)
        CUDA.CUBLAS.gemm!('N', 'N', one(T), V, u, zero(T), v)
        eltype(s).(clamp01.(reshape(colorview(base_color_type(U){T}, v), size(s))))
    end
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" replace_normalized_principle_component(
      p :: NormalizedColorPCAState{T}
    , s :: CUDA.CuArray{U}
    , η :: CUDA.CuArray{T}
    ) where T <: AbstractFloat where U <: Color3 = begin
        broadcast!( (x, ma, mi)->x*(ma-mi)+mi, η, η, p.ma, p.mi)
        d_V = CUDA.CuArray{T}(p.V)
        s0 = synthesis(d_V,s,η)
    end
end

end
