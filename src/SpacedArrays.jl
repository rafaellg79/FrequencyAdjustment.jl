module SpacedArrays

using Lazy
using OffsetArrays
using MappedArrays

export SpacedArray
export SpacedRange
export ParentIndex
export ParentIndices

export sharedindices
export SpacedCartesianIndices

struct ParentIndex <: Signed
    value::Int
end

const ParentIndices = UnitRange{ParentIndex}

struct SpacedRange{T,S} <: OrdinalRange{T,S}
    steprange :: StepRange{T,S}
end

struct SpacedArray{T,N,AA<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::AA
    offsets::NTuple{N,Int}
    steps::NTuple{N,Int}
end

function SpacedArray(A::AbstractArray{T,N}, inds::Vararg{StepRange,N}) where {T,N}
    lA = map(length, axes(A))
    lI = map(length, inds)
    lA == lI || throw(DimensionMismatch("supplied axes do not agree with the size of the array (got size $lA for the array and $lI for the indices"))
    offsets = getproperty.(inds, :start) .- 1
    steps = getproperty.(inds, :step)
    SpacedArray(A, offsets, steps)
end

SpacedArray(A::AbstractArray{T,N}, inds::NTuple{N,StepRange}) where {T,N} = SpacedArray(A, inds...)
SpacedArray(A::AbstractArray{T,N}, inds::Vararg{SpacedRange,N}) where {T,N} = SpacedArray(A, getfield.(inds, :steprange))
SpacedArray(A::AbstractArray{T,N}, inds::NTuple{N,SpacedRange}) where {T,N} = SpacedArray(A, getfield.(inds, :steprange))

function Base.LinearIndices(axes::NTuple{N,SpacedRange}) where N
    # SpacedArray(Base.LinearIndices(map(length,srs)), srs)
    lens = map(length, axes)
    SpacedArray(mappedarray(ParentIndex, Base.LinearIndices(lens)), axes)
end

function OffsetArrays.OffsetArray(val::V, S::SpacedArray{T,N}) where {V,T,N}
    lens = map(axes(S)) do sr::SpacedRange
        first(sr):last(sr)
    end
    A = OffsetArrays.OffsetArray(Array{Union{V,T}, N}(fill(val, length.(lens)...)), lens...)
    for ind in SpacedCartesianIndices(S)
        A[ind] = S[ind]
    end
    A
end

Base.size(A::SpacedArray) = size(A.parent)

using Base: @propagate_inbounds

Lazy.@forward SpacedRange.steprange (
    Base.first, Base.step, Base.last, Base.size, Base.length
)
Base.show(io::IO, s::SpacedRange) = print(io, "SpacedRange(", sprint(show, s.steprange), ")")

struct SpacedCartesianIndices{N}
    it::Iterators.ProductIterator{NTuple{N,T}} where T
end

SpacedCartesianIndices(inds::NTuple{N,SpacedRange}) where N = SpacedCartesianIndices(Iterators.product(inds...))
SpacedCartesianIndices(S::SpacedArray) where N = SpacedCartesianIndices(axes(S))

function _iterate_to_cartesian(ret)
    if isnothing(ret)
        return nothing
    else
        data, st = ret
        return (CartesianIndex(data), st)
    end
end

function Base.iterate(sci::SpacedCartesianIndices)
    _iterate_to_cartesian( iterate(sci.it) )
end
function Base.iterate(sci::SpacedCartesianIndices, st)
    _iterate_to_cartesian( iterate(sci.it, st) )
end
Base.length(sci::SpacedCartesianIndices) = length(sci.it)
Base.size(sci::SpacedCartesianIndices) = size(sci.it)
Base.eltype(::Type{SpacedCartesianIndices{N}}) where N = CartesianIndex{N}

Base.IteratorSize(::SpacedCartesianIndices{N}) where N = Base.HasShape{N}()

function sharedindices(arrays::AbstractArray...)
    SpacedCartesianIndices(Iterators.product(map(intersect, map(axes, arrays)...)...))
end

Base.IndexStyle(::Type{SpacedArray}) = IndexCartesian()
Base.IndexStyle(::SpacedArray) = IndexCartesian()

Base.eachindex(::IndexCartesian, S::SpacedArray) = SpacedCartesianIndices(axes(S))

function Base.similar(S::SpacedArray, ::Type{T}, inds::Tuple{SpacedRange,Vararg{SpacedRange}}) where T
    B = similar(S.parent, T, map(length, inds))
    SpacedArray(B, inds...)
end

Base.to_shape(sr::SpacedRange) = sr
Base.to_shape(srs::NTuple{N,SpacedRange}) where N = map(Base.to_shape, srs)

_reshape(::SpacedArray, params...) =
    error("Cannot reshape SpacedArray S. Please use S.parent.")

Base.reshape(S::SpacedArray, dims::Union{Colon,Int}...) = _reshape(S, dims)
Base.reshape(S::SpacedArray, dims::Int...) = _reshape(S, dims)

Base.view(S::SpacedArray, dims::Vararg{Any,N}) where N =
    error("Cannot get view of a SpacedArray S. Please use S.parent.")

ParentIndex(x::ParentIndex) = x

Base.show(io::IO, x::ParentIndex) = print(io, "ParentIndex(", x.value, ")")
Base.Int64(x::ParentIndex) = x
Base.:(<)(x::Real, y::ParentIndex) = Base.:(<)(x, y.value)
Base.:(<)(x::ParentIndex, y::Real) = Base.:(<)(x.value, y)
Base.:(<)(x::ParentIndex, y::ParentIndex) = Base.:(<)(x.value, y.value)

Base.:(<=)(x::Real, y::ParentIndex) = Base.:(<=)(x, y.value)
Base.:(<=)(x::ParentIndex, y::Real) = Base.:(<=)(x.value, y)
Base.:(<=)(x::ParentIndex, y::ParentIndex) = Base.:(<=)(x.value, y.value)

Base.:(==)(x::Real, y::ParentIndex) = Base.:(==)(x, y.value)
Base.:(==)(x::ParentIndex, y::Real) = Base.:(==)(x.value, y)
Base.:(==)(x::ParentIndex, y::ParentIndex) = Base.:(==)(x.value, y.value)

Base.:(+)(x::ParentIndex, y::ParentIndex) = ParentIndex(Base.:(+)(x.value, y.value))
Base.:(-)(x::ParentIndex, y::ParentIndex) = ParentIndex(Base.:(-)(x.value, y.value))
Base.:(*)(x::ParentIndex, y::ParentIndex) = ParentIndex(Base.:(*)(x.value, y.value))
Base.:rem(x::ParentIndex, y::ParentIndex) = ParentIndex(Base.:rem(x.value, y.value))

Base.:(+)(x::Integer, y::ParentIndex) = ParentIndex(Base.:(+)(x, y.value))
Base.:(-)(x::Integer, y::ParentIndex) = ParentIndex(Base.:(-)(x, y.value))
Base.:(*)(x::Integer, y::ParentIndex) = ParentIndex(Base.:(*)(x, y.value))
Base.:rem(x::Integer, y::ParentIndex) = ParentIndex(Base.:rem(x, y.value))

Base.:(+)(x::ParentIndex, y::Integer) = ParentIndex(Base.:(+)(x.value, y))
Base.:(-)(x::ParentIndex, y::Integer) = ParentIndex(Base.:(-)(x.value, y))
Base.:(*)(x::ParentIndex, y::Integer) = ParentIndex(Base.:(*)(x.value, y))
Base.:rem(x::ParentIndex, y::Integer) = ParentIndex(Base.:rem(x.value, y))

Base.:(-)(x::ParentIndex) = ParentIndex(Base.:(-)(x.value))

Base.length(s::ParentIndices) = Base.unsafe_length(s).value

Base.UnitRange(s::SpacedRange) = ParentIndices(range(1; length=length(s.steprange)))
Base.UnitRange{T}(s::SpacedRange) where T = ParentIndices(range(1; length=length(s.steprange)))

Base.parent(S::SpacedArray) = S.parent
function Base.axes(S::SpacedArray{T,N}) where {T,N}
    ntuple(N) do i
        SpacedRange(range(S.offsets[i] .+ 1; length=size(S,i), step=S.steps[i]))
    end
end

function Base.showarg(io::IO, a::SpacedArray, toplevel)
    print(io, "SpacedArray(")
    Base.showarg(io, parent(a), false)
    print(io, ')')
    toplevel && print(io, " with eltype ", eltype(a))
end

@inline function inrange(i::Int, start::Int, step::Int, len::Int)
    return i >= start &&
        (i - start) ÷ step < len &&
        rem(i - start, step) == 0
end

@inline parentindex(i::Int, start::Int, step::Int) = 1 + ((i - start) ÷ step)
@inline parentindex(c::Colon, start::Int, step::Int) = c
@inline @propagate_inbounds function parentindex(S::SpacedArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(S, I...)
    return parentindex.(I, S.offsets .+ 1, S.steps)
end
@inline @propagate_inbounds function parentindex(S::SpacedArray{T,N}, I::CartesianIndex{N}) where {T,N}
    return CartesianIndex( parentindex(S, I.I...) )
end

@inline @propagate_inbounds function nearestvalid(S::SpacedArray{T,N}, I::Vararg{Int,N}) where {T,N}
    ntuple(N) do i
        1 + S.offsets[i] + div(I[i] - S.offsets[i], S.steps[i]) * S.steps[i]
    end
end
@inline @propagate_inbounds function nearestvalid(S::SpacedArray{T,N}, I::CartesianIndex{N}) where {T,N}
    return CartesianIndex( nearestvalid(S, I.I...) )
end

@inline @propagate_inbounds function spacedstep(S::SpacedArray{T,N}, I::NTuple{N}, n::Vararg{Int,N}) where {T,N}
    ntuple(N) do i
        I[i] + n[i] * S.steps[i]
    end
end

@inline @propagate_inbounds function spacedstep(S::SpacedArray{T,N}, I::CartesianIndex{N}, n::Vararg{Int,N}) where {T,N}
    CartesianIndex( spacedstep(S, I.I, n...) )
end

@inline function Base.checkindex(::Type{Bool}, sr::SpacedRange, I::Integer)
    inrange(I, first(sr.steprange), step(sr.steprange), length(sr.steprange))
end
@inline function Base.checkindex(::Type{Bool}, sr::SpacedRange, c::Colon)
    return true
end

@inline @propagate_inbounds function Base.getindex(S::SpacedArray{T,N}, I::Vararg{ParentIndex,N}) where {T,N}
    return S.parent[ getfield.(I, :value)... ]
end
@inline @propagate_inbounds function Base.getindex(S::SpacedArray{T,N}, i::ParentIndex) where {T,N}
    return S.parent[ i.value ]
end
@inline @propagate_inbounds function Base.getindex(S::SpacedArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(S, I...)
    return S.parent[ parentindex.(I, S.offsets .+ 1, S.steps)... ]
end
@inline @propagate_inbounds function Base.getindex(S::SpacedArray{T,N}, I::Vararg{Union{Int,Colon},N}) where {T,N}
    @boundscheck checkbounds(S, I...)
    new_inds = ntuple(N) do n
        I[n] isa Int ? missing : axes(S)[n]
    end
    new_inds = Tuple(collect(skipmissing(new_inds)))
    return SpacedArray( S.parent[ parentindex.(I, S.offsets .+ 1, S.steps)... ], new_inds )
end
@inline @propagate_inbounds function Base.getindex(S::SpacedArray{T,N}, I::CartesianIndex) where {T,N}
    getindex(S, I.I...)
end
@inline @propagate_inbounds function Base.getindex(S::SpacedArray{T,N}, i::Int) where {T,N}
    error("SpacedArray S cannot be accessed linearly. Please access S.parent[$i].")
end
@inline @propagate_inbounds function Base.getindex(S::SpacedArray{T,N}, I...) where {T,N}
    error("Indexing of SpacedArray with index type $(typeof(I)) not implemented.")
end

@inline @propagate_inbounds function Base.setindex!(S::SpacedArray{T,N}, val, I::Vararg{ParentIndex,N}) where {T,N}
    S.parent[ getfield.(I, :value)... ] = val
    val
end
@inline @propagate_inbounds function Base.setindex!(S::SpacedArray{T,N}, val, i::ParentIndex) where {T,N}
    S.parent[ i.value ] = val
    val
end
@inline @propagate_inbounds function Base.setindex!(S::SpacedArray{T,N}, val, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(S, I...)
    S.parent[ parentindex.(I, S.offsets .+ 1, S.steps)... ] = val
    val
end
@inline @propagate_inbounds function Base.setindex!(S::SpacedArray{T,N}, val, I::CartesianIndex) where {T,N}
    setindex!(S, val, I.I...)
end
@inline @propagate_inbounds function Base.setindex!(S::SpacedArray{T,N}, val, i::Int) where {T,N}
    error("SpacedArray S cannot be accessed linearly. Please access S.parent[$i].")
end

end
