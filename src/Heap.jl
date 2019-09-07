
using DataStructure: BinaryMinHeap

mutable struct Heap{T}
        h::BinaryMinHeap{T}
        maxsz::Int
end

Heap{T}(maxsz::Int) = Heap{T}(BinaryMinHeap{T}(), maxsz)

import Base.length
length(h::Heap{T}) where {T} = length(h.h)

import Base.isempty
isempty(h::Heap{T}) where {T} = isempty(h.h)

import Base.push!
function push!(h::Heap{T}, v::T)
        if length(h) == h.maxsz && top(h.h) < v; pop!(h.h); end
        push!(h.h, v)
end

import Base.empty!
empty!(h::Heap{T}) = empty!(h.h.valtree)



