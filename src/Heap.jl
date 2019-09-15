
using DataStructures: BinaryMinHeap

mutable struct Heap{T}
        h::BinaryMinHeap{T}
        maxsz::Int
end

Heap{T}(maxsz::Int) where {T} = Heap{T}(BinaryMinHeap{T}(), maxsz)

import Base.length
length(h::Heap{T}) where {T} = length(h.h)

import Base.isempty
isempty(h::Heap{T}) where {T} = isempty(h.h)

import Base.push!
function push!(h::Heap{T}, v::T) where {T}
        if length(h) == h.maxsz && top(h.h) < v; pop!(h.h); end
        if length(h) < h.maxsz; push!(h.h, v); end
end

import Base.empty!
empty!(h::Heap{T}) where {T} = empty!(h.h.valtree)

import Base.iterate
iterate(h::Heap{T}, i=1) where {T} = i > length(h) ? nothing : (h.h.valtree[i], i + 1)





